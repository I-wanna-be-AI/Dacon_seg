from PIL import Image
import os
import cv2
import pandas as pd
from glob import glob
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from .augmen import *
from .rle import *

class customDataset(Dataset):
    def __init__(self, X, transform,  y = None):
        self.X = X
        self.y = y
        self.transform = transform

    def __getitem__(self, index):
        impath = self.X[index]
        # cv2_img = cv2.imread(impath, cv2.IMREAD_COLOR)
        cv2_img = Image.open(impath).convert('RGB')
        # img = Image.open(impath).convert('RGB')
        img = self.transform(cv2_img)

        if self.y is not None:
            target = self.y[index]
            return img, target
        else:
            fname = os.path.basename(impath)
            return img, fname

    def __len__(self, ):
        return len(self.X)


class SatelliteDataset(Dataset):
    def __init__(self, csv_file, transform=None, infer=False):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.infer = infer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 1]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.infer:
            if self.transform:
                image = self.transform(image=image)['image']
            return image

        mask_rle = self.data.iloc[idx, 2]
        mask = rle_decode(mask_rle, (image.shape[0], image.shape[1]))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask


