# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import csv
import math
from loader import SketchDataSet
import argparse
from utils import VisdomPlotter
import pandas as pd


# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

regress_data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, "model/best_resnet.pth")

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def regress_train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    plotter = VisdomPlotter.VisdomLinePlotter(env_name='main')

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    #_, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                #running_corrects += torch.sum(preds == labels.data)
                # running_corrects += (1.0 - math.sqrt((torch.mean(torch.pow(outputs-labels, 2))))) / 1.0

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = 1 - epoch_loss

            plotter.plot('loss', phase, 'MSE Loss', epoch, epoch_loss)
            plotter.plot('acc', phase, 'Regression Acc', epoch, epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, "model/best_resnet.pth")

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    plt.show(block=True)
                    model.train(mode=was_training)
                    return
        plt.show(block=True)
        model.train(mode=was_training)

from matplotlib.pyplot import cm

def regression_visualize_model(model, num_images=6):
    model.eval();
    curr_img_cnt = 0

    color = iter(cm.rainbow(np.linspace(0, 1, num_images)))

    with torch.no_grad():
        for i, (inputs, ground_truth) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            #grounnd_truth = outputs.to(device)

            outputs = model(inputs).cpu().detach().numpy()
            ground_truth = ground_truth.cpu().detach().numpy()


            x = np.arange(outputs.shape[1])

            for j in range(inputs.size()[0]):
                c = next(color)
                c = np.reshape(c, -1)
                curr_img_cnt += 1
                ax = plt.subplot(num_images//4, 4, curr_img_cnt)
                # plt.scatter(x, outputs[j], s=20, marker='o', c=c)
                # plt.scatter(x, ground_truth[j], s=20, marker='^', c=c)
                l1 = plt.plot(x, outputs[j], '.r-', label='predicted')
                l2 = plt.plot(x, ground_truth[j], 'xb-', label='ground truth')
                # l1 = plt.plot(x, outputs[j], '.r-')
                # l2 = plt.plot(x, ground_truth[j], 'xb-')

                if curr_img_cnt == num_images:
                    plt.figlegend(loc='upper left', ncol=1)
                    plt.show(block=True)
                    return


def regression_test(model):
    model.eval();

    arr = []
    with torch.no_grad():
        for i, (files, inputs) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            outputs = model(inputs).cpu().detach().numpy()

            for j in range(inputs.size()[0]):
                csv_row = outputs[j].tolist()
                csv_row.insert(0, os.path.basename(files[j]))
                arr.append(csv_row)

    a = np.array(arr)
    pd.DataFrame(a).to_csv("output.csv", header=None, index=None)


def create_labels_from_csv(filename):
    line_cnt = 0
    with open(filename, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        rows = []
        for row in csv_reader:
            if (line_cnt == 0):
                print(f'Column names :  {" ".join(row)}')
            else:
                rows.append(row)
                # print(row)
            line_cnt += 1
        y = np.asarray(rows);
        y = np.delete(y, 16, 1)
        print(y.shape)
        return y

def classify():
    data_dir = 'data/hymenoptera_data'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    plt.ion()   # interactive mode
    #imshow(out, title=[class_names[x] for x in classes])

    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 16)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=25)

    # model_ft.load_state_dict(torch.load('model/best_resnet.pth'))
    # visualize_model(model_ft);

def regress(should_train=False, should_test=True):
    global dataloaders, dataset_sizes, device
    # data_dir = 'data/sketch-gen'
    image_datasets = {x: SketchDataSet.SketchDataSet('curve_params.csv', os.path.join(data_dir, x),
                                              regress_data_transforms[x])
                      for x in ['train', 'val']}
    image_datasets['test'] = SketchDataSet.SketchTestDataSet(os.path.join(test_data_dir), regress_data_transforms['val'])

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].curve_param_names

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    #plt.ion()   # interactive mode
    #imshow(out, title=[class_names for x in classes])

    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 16)

    model_ft = model_ft.to(device)

    if (should_train):
    #criterion = nn.CrossEntropyLoss()
        criterion = nn.MSELoss()

        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        model_ft = regress_train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                               num_epochs=250)
    elif should_test:
        model_ft.load_state_dict(torch.load('model/best_resnet.pth'))
        regression_test(model_ft)
    else:
        model_ft.load_state_dict(torch.load('model/best_resnet.pth'))
        regression_visualize_model(model_ft, 24)


if __name__ == '__main__':
    global data_dir, test_data_dir
    parser = argparse.ArgumentParser();
    parser.add_argument('--data_dir', type=str, default='data/sketchgen')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    # parser.add_argument('--train', nargs='?', type=bool, const=False, default=True)
    args = parser.parse_args()

    data_dir = args.data_dir
    test_data_dir = os.path.join(data_dir, 'test')
    should_train = args.train

    regress(should_train)

    #create_labels_from_csv('data/sketch-gen/params.csv')
    #exit(0)
