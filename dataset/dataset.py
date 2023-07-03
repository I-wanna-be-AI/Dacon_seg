from PIL import Image
import os
import cv2
import pandas as pd
from glob import glob
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from .augmen import *

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

def get_dataloader(args):
    train_aug, valid_aug = get_aug(args)
    fake_images = glob("data/train/fake_images/*.png")
    real_images = glob("data/train/real_images/*.png")
    labels = [1] * len(fake_images) + [0] * len(real_images)
    X_train, X_val, Y_train, Y_val = train_test_split(fake_images + real_images, labels, test_size= args.split, random_state = args.seed, shuffle=True)
    
    train_dataset = customDataset(X = X_train, y = Y_train, transform = train_aug)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)                                 
    train_dl = DataLoader(train_dataset, batch_size = args.batchsize, num_workers = 4 , shuffle = False, sampler = train_sampler, pin_memory = True)

    val_dataset = customDataset(X = X_val, y = Y_val, transform = valid_aug)
    valid_dl = DataLoader(val_dataset, batch_size = args.batchsize, num_workers = 4, shuffle = False)
    
    return train_dl, valid_dl, train_sampler


def get_testloader(args):
    X_test = glob(f"data/test/images/*.png")
    test_df = pd.read_csv("data/sample_submission.csv")
    _, valid_aug = get_aug(args)
    test_dataset = customDataset(X = X_test, transform = valid_aug)
    test_dl = DataLoader(test_dataset, batch_size = args.batchsize, num_workers = 4, shuffle = False)

    return test_dl, test_df