import torch

torch.cuda.current_device()
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
import os
from loader import SketchDataSet
from PIL import Image


def get_resnet_model(resnet_type, pretrained):
    if resnet_type == '101':
        model_ft = models.resnet101(pretrained=pretrained)
    elif resnet_type == '18':
        model_ft = models.resnet18(pretrained=pretrained)
    elif resnet_type == '50':
        model_ft = models.resnet50(pretrained=pretrained)
    else:
        raise Exception("Unknown resenet type", resnet_type)
    return model_ft


class ClassifierInfer:
    def __init__(self, data_dir, resnet_type):
        self.resnet_type = resnet_type
        self.class_names = None
        self.model_ft = None
        self.data_dir = data_dir
        self.save_name = os.path.basename(data_dir)
        self.device = None
        self.regress_data_transforms = {
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
            # straight away resize to 224
            'infer':transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

    def load_model(self):
        image_datasets = {x: datasets.ImageFolder(os.path.join(self.data_dir, x),
                                                  self.regress_data_transforms[x])
                          for x in ['train', 'val']}

        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                      shuffle=True, num_workers=4)
                       for x in ['train', 'val']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.class_names = image_datasets['train'].classes

        # Get a batch of training data
        inputs, classes = next(iter(dataloaders['train']))

        # Make a grid from batch
        out = torchvision.utils.make_grid(inputs)

        # plt.ion()   # interactive mode
        # imshow(out, title=[class_names for x in classes])

        model_ft = get_resnet_model(self.resnet_type, False)
        num_ftrs = model_ft.fc.in_features

        # setting number of params
        num_classes = len(self.class_names)
        print("class size : ", num_classes, self.class_names)
        output_layer_size = num_classes
        print("Setting output layer size to : ", output_layer_size)
        model_ft.fc = nn.Linear(num_ftrs, output_layer_size)

        model_ft = model_ft.to(device)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_ft = model_ft.to(self.device)
        model_name = os.path.join('model', self.save_name + '.pth');
        self.model_ft.load_state_dict(torch.load(model_name))
        self.model_ft.eval();

    def convert_output_id_to_class_name(self, outputs):
        _, preds = torch.max(outputs, 1)

        predicted_class_names = []
        for j in range(len(preds)):
            pred_class_name = self.class_names[preds[j]]
            predicted_class_names.append(pred_class_name)
        return predicted_class_names

    # def infer_imgs(self, img_numpy_arrs):
    #     input_list = []
    #     num_imgs = img_numpy_arrs.shape[0]
    #     for i in range(0, num_imgs):
    #         img_numpy = img_numpy_arrs[i]
    #         sample = Image.fromarray(img_numpy);
    #         input = self.regress_data_transforms['val'](sample)
    #         input_list.append(input)
    #         sample.save('test' + str(i) + '.png');
    #     inputs = torch.stack(input_list, dim=0)
    #     inputs = inputs.to(self.device)
    #     outputs = self.model_ft(inputs).cpu().detach().numpy()
    #     return outputs

    def infer_imgs_raw(self, img_tensors):
        inputs = img_tensors.to(self.device)
        outputs = self.model_ft(inputs)
        pred_class_names = self.convert_output_id_to_class_name(outputs)
        return pred_class_names

    # def infer_img(self, img_numpy):
    #     sample = Image.fromarray(img_numpy);
    #     sample.save('test.png');
    #     inputs = self.regress_data_transforms['val'](sample)
    #     inputs = inputs.unsqueeze(0)
    #     inputs = inputs.to(self.device)
    #     outputs = self.model_ft(inputs).cpu().detach().numpy()
    #     return outputs[0]

    # def infer_tensor(self, img_tensor):
    #     inputs = img_tensor.unsqueeze(0)
    #     inputs = img_tensor.to(self.device)
    #     outputs = self.model_ft(inputs).cpu().detach().numpy()
    #     return outputs[0]


class RegressInfer:
    def __init__(self, data_dir, resnet_type):
        self.model_ft = None
        self.data_dir = data_dir
        self.save_name = os.path.basename(data_dir)
        self.resnet_type = resnet_type
        self.device = None
        self.regress_data_transforms = {
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

    def load_model(self):
        image_datasets = {x: SketchDataSet.SketchDataSet('curve_params.csv', os.path.join(self.data_dir, x),
                                                         self.regress_data_transforms[x])
                          for x in ['train']}

        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                      shuffle=True, num_workers=4)
                       for x in ['train']}

        inputs, curve_params = next(iter(dataloaders['train']))
        # model_ft = models.resnet18(pretrained=True)
        model_ft = get_resnet_model(self.resnet_type, False)
        num_ftrs = model_ft.fc.in_features

        # setting number of params
        print("Curve parameters size : ", curve_params.shape)
        output_layer_size = curve_params.shape[1]
        print("Setting output layer size to : ", output_layer_size)
        model_ft.fc = nn.Linear(num_ftrs, output_layer_size)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_ft = model_ft.to(self.device)
        model_name = os.path.join('model', self.save_name + '.pth');
        self.model_ft.load_state_dict(torch.load(model_name))
        self.model_ft.eval();

    def infer_imgs(self, img_numpy_arrs):
        input_list = []
        num_imgs = img_numpy_arrs.shape[0]
        for i in range(0, num_imgs):
            img_numpy = img_numpy_arrs[i]
            sample = Image.fromarray(img_numpy);
            input = self.regress_data_transforms['val'](sample)
            input_list.append(input)
            sample.save('test' + str(i) + '.png');
        inputs = torch.stack(input_list, dim=0)
        inputs = inputs.to(self.device)
        outputs = self.model_ft(inputs).cpu().detach().numpy()
        return outputs

    def infer_img(self, img_numpy):
        sample = Image.fromarray(img_numpy);
        sample.save('test.png');
        inputs = self.regress_data_transforms['val'](sample)
        inputs = inputs.unsqueeze(0)
        inputs = inputs.to(self.device)
        outputs = self.model_ft(inputs).cpu().detach().numpy()
        return outputs[0]

    def infer_raw(self, img_tensor):
        inputs = img_tensor.unsqueeze(0)
        inputs = inputs.to(self.device)
        outputs = self.model_ft(inputs).cpu().detach().numpy()
        return outputs[0]


class TotalInfer:
    def __init__(self, base_dir, classifer_dir, regress_dirs, resnet_type):
        self.data_dirs = regress_dirs
        self.regress_model_map = {}
        for regress_dir in regress_dirs:
            regress_infer = RegressInfer(os.path.join(base_dir, regress_dir), resnet_type)
            regress_infer.load_model()
            self.regress_model_map[regress_dir] = regress_infer

        self.classifier_infer = ClassifierInfer(os.path.join(base_dir, classifer_dir), resnet_type)
        self.classifier_infer.load_model()

        print(self.classifier_infer, self.regress_model_map)

        self.regress_data_transforms = {
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

    def infer_imgs(self, img_numpy_arrs):
        input_list = []
        num_imgs = img_numpy_arrs.shape[0]
        for i in range(0, num_imgs):
            img_numpy = img_numpy_arrs[i]
            sample = Image.fromarray(img_numpy);
            input = self.regress_data_transforms['val'](sample)
            input_list.append(input)
            sample.save('test' + str(i) + '.png');
        inputs = torch.stack(input_list, dim=0)
        classifer_outputs = self.classifier_infer.infer_imgs_raw(inputs)

        regress_outputs = []
        for i in range(0, num_imgs):
            assert (type(classifer_outputs[i]) == str)
            data_type = classifer_outputs[i]
            print('img ', i, ' is ', data_type)
            regress_output = self.regress_model_map[data_type].infer_raw(input_list[i])
            regress_outputs.append({data_type: regress_output.tolist()})

        return regress_outputs
