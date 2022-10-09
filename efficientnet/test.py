from email.mime import image
import os
import easydict
import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import cv2

import torch
from torch.utils.data import DataLoader, Dataset
from efficientnet_pytorch import EfficientNet
import torch_xla.core.xla_model as xm

import albumentations
import albumentations.pytorch

device = 'cpu'

class DatasetMNIST(torch.utils.data.Dataset):
    def __init__(self, image_folder, label_df, transforms):        
        self.image_folder = image_folder
        self.imgs_arr = sorted(os.listdir(image_folder))
        self.label_df = label_df
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs_arr)
    
    def __getitem__(self, index):        
        image_fn = os.path.join(self.image_folder, self.imgs_arr[index])                                              
        image = cv2.imread(image_fn, cv2.IMREAD_GRAYSCALE)
        label = self.label_df.iloc[index,1:].values.astype('float')

        if self.transforms:            
            image = self.transforms(image=image)['image'] / 255.0

        return image, label
    

mnist_transforms = {
    'train' : albumentations.Compose([
            albumentations.RandomRotate90(),
            albumentations.OneOf([
                albumentations.GridDistortion(distort_limit=(-0.3, 0.3), border_mode=cv2.BORDER_CONSTANT, p=1),
                albumentations.ShiftScaleRotate(rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, p=1),        
                albumentations.ElasticTransform(alpha_affine=10, border_mode=cv2.BORDER_CONSTANT, p=1),
            ], p=1),    
            albumentations.Cutout(num_holes=16, max_h_size=15, max_w_size=15, fill_value=0),
            albumentations.pytorch.ToTensorV2(),
        ]),
    'valid' : albumentations.Compose([        
        albumentations.pytorch.ToTensorV2(),
        ]),
    'test' : albumentations.Compose([        
        albumentations.pytorch.ToTensorV2(),
        ]),
}

path = '/home/nora/project/rsc/dirty_mnist'
label_path = '/home/nora/project/rsc/dirty_mnist_2nd_answer.csv'
label = pd.read_csv(label_path)
batch_size = 128

ds = DatasetMNIST(path, label, mnist_transforms['train'])
loader = DataLoader(ds, batch_size)

def train(loader, model, loss_fc, optimizer, scheduler=None):
    device = 'cpu'
    sw = True
    
    size = 0
    running_loss = .0
    running_corrects = 0

    epoch_loss = .0
    epoch_acc = 0.0

    model.train()
    for idx, (X, y) in tqdm(enumerate(loader, 1)):
        X, y = X.to(device), y.to(device)
        
        output = model(X)
        loss = loss_fc(output, y)
        
        if sw:
            sw = False
            print(loss.size())
            print(loss)
        
        size += len(y)
        running_loss += loss.cpu().item() * len(y)

        epoch_loss = running_loss / float(n)

        output = output > 0.5
        running_corrects += (output == train_y).sum()
        epoch_acc = running_corrects / train_y.size(1) / n        

        optimizer.zero_grad()
        loss.backward()
        break
    print('Train Loss')
    print('loss - {:.5f}, acc - {:.5f}'.format(epoch_loss, epoch_acc))

    if scheduler:
        scheduler.step(epoch_loss)

    return epoch_loss, epoch_acc

model = EfficientNet.from_name("efficientnet-b8", in_channels=1, num_classes=26, dropout_rate=0.5)
criterion = torch.nn.MultiLabelSoftMarginLoss()
optim = torch.optim.Adam(model.parameters(), lr=1e-4)
for X, y in loader:
    train(loader, model, criterion, optim)
    break
    