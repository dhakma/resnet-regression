import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
from PIL import Image
import os


class SketchDataSet(Dataset):

    def __init__(self, csv_file_name, root_dir, transform=None):
        self.curve_params = pd.read_csv(os.path.join(root_dir, csv_file_name))
        self.root_dir = root_dir
        self.transform = transform
        self.curve_param_names = list(self.curve_params.columns)

    def __len__(self):
        return len(self.curve_params)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.curve_params.iloc[idx, 0]);
        # image = io.imread(img_name);
        sample = Image.open(img_name);
        col_len = len(self.curve_params.iloc[idx])
        final_col_len = col_len
        curve_params = self.curve_params.iloc[idx, 1:final_col_len].values
        #curve_params = curve_params.astype('float').reshape(-1, col_len - 1)
        curve_params = curve_params.astype('float32')

        if self.transform:
            sample = self.transform(sample)
        return sample, curve_params

    def get_output_params(self):
        return self.curve_params


class SketchTestDataSet(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        img = Image.open(file)
        if self.transform:
            img = self.transform(img)

        return file, img
