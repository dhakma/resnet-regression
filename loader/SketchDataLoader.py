import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import os

class SketchDataLoader(Dataset):

    def __init__(self, csv_file_name, root_dir, transform=None):
        self.curve_params = pd.read_csv(csv_file_name)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.curve_params)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.curve_params.iloc[idx, 0]);
        image = io.imread(img_name);
        col_len = len(self.curve_params.iloc[idx])
        curve_params = self.curve_params.iloc[idx, 1:(col_len - 1)]
        curve_params = curve_params.reshape(-1, col_len - 1)
        sample = {'image' : image, 'curve_params' : curve_params}
        if (self.transform):
            sample = self.transform(sample)
        return sample


