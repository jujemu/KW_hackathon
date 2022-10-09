import albumentations as A
import albumentations.pytorch as AP

import os
import random
import sys
import cv2
import tqdm

import numpy as np
import torch
import torch_xla.core.xla_model as xm
from torch.utils.data import Dataset


def get_aug():
    transform = {
        'train': A.Compose([
            A.Rotate(limit=50),
            A.OneOf([
                A.GridDistortion(distort_limit=(-0.3, 0.3),
                                 border_mode=cv2.BORDER_CONSTANT, p=1),
                A.ShiftScaleRotate(
                    rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, p=1),
                A.ElasticTransform(
                    alpha_affine=10, border_mode=cv2.BORDER_CONSTANT, p=1),
            ], p=1),
            A.Cutout(num_holes=24, max_h_size=10, max_w_size=10, fill_value=0),
            AP.ToTensorV2(),
        ]),
        'valid': A.Compose([
            AP.ToTensorV2(),
        ])
    }
    return transform


class DatasetMNIST(Dataset):
    def __init__(self, image_folder, label_df, transforms):
        self.image_folder = image_folder
        self.label_df = label_df
        self.transforms = transforms

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, index):
        image_fn = self.image_folder +\
            str(self.label_df.iloc[index, 0]).zfill(5) + '.png'

        image = cv2.imread(image_fn, cv2.IMREAD_GRAYSCALE)
        image = image.reshape([256, 256, 1])

        label = self.label_df.iloc[index, 1:].values.astype('float')

        if self.transforms:
            image = self.transforms(image=image)['image'] / 255.0

        return image, label


def train(train_loader, model, loss_func, args, optimizer, scheduler=None):
    n = 0
    running_loss = 0.0
    running_corrects = 0

    epoch_loss = 0.0
    epoch_acc = 0.0

    model.train()
    with tqdm.tqdm(train_loader, total=len(train_loader), desc="Train", file=sys.stdout) as iterator:
        for train_x, train_y in iterator:
            train_x = train_x.to(args.device)
            train_y = train_y.to(args.device)

            output = model(train_x)

            loss = loss_func(output, train_y)

            n += train_x.size(0)
            running_loss += loss.item() * train_x.size(0)

            epoch_loss = running_loss / float(n)

            output = output > 0.42
            running_corrects += (output == train_y).cpu().sum().item()
            epoch_acc = running_corrects / train_y.size(1) / n

            log = 'loss - {:.5f}, acc - {:.5f}'.format(epoch_loss, epoch_acc)

            iterator.set_postfix_str(log)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            xm.mark_step()

    if scheduler:
        scheduler.step(epoch_loss)

    return epoch_loss, epoch_acc


def validate(valid_loader, model, loss_func, args):
    n = 0
    running_loss = 0.0
    running_corrects = 0

    epoch_loss = 0.0
    epoch_acc = 0.0

    model.eval()
    with tqdm.tqdm(valid_loader, total=len(valid_loader), desc="Valid", file=sys.stdout) as iterator:
        for train_x, train_y in iterator:
            train_x = train_x.to(args.device)
            train_y = train_y.to(args.device)

            with torch.no_grad():
                output = model(train_x)

            loss = loss_func(output, train_y)

            n += train_x.size(0)
            running_loss += loss.item() * train_x.size(0)

            epoch_loss = running_loss / float(n)

            output = output > 0.42
            running_corrects += (output == train_y).cpu().sum().item()
            epoch_acc = running_corrects / train_y.size(1) / n

            log = 'loss - {:.5f}, acc - {:.5f}'.format(epoch_loss, epoch_acc)

            iterator.set_postfix_str(log)

    return epoch_loss, epoch_acc


def infer(loader, model, args, sample):
    print('=' * 50)
    print('Inference has been started')
    
    init = 0
    model.eval()
    with torch.no_grad():
        for X, _ in tqdm.tqdm(loader, total=len(loader)):
            X = X.to(args.device)
            output = model(X)
            output = output.cpu().numpy()
            
            gap = X.size(0)
            sample.iloc[init:init+gap, 1:] += output            
            init += gap
    print()
    print('=' * 50)
    print('Inference is completed')
    print()
    return sample


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    