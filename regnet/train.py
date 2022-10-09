import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import albumentations as A
import albumentations.pytorch as AP
import click
import cv2
import easydict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import tqdm

import regnet_pytorch.regnet as regnet

@click.command()
@click.pass_context
@click.option('--network', 'network', default=None, help='pretrained model weight')
@click.option('--epochs', 'epochs', type=int, default=20, help='train for epochs')
@click.option('--kfold', 'kfold', type=int, help='only one fold is used in training', required=True)
def main(
    ctx: click.Context,
    network: str,
    epochs: int,
    kfold: int
):
    """
    Train regnet model, using TPU in gcp.

    input path : ../rsc

    checkpoint path : ./checkpoint

    model : "regnet"

    Ex>  

    \t
    python3 train.py --kfold 0

    \t
    python3 train.py --network checkpoint/2022-09-25_0614/model_best.pth --kfold 0
    """

    device = xm.xla_device()
    if network is None:
        model = Network(init=True)
    else:
        model = Network()
        model.load_state_dict(torch.load(network))
    model = model.to(device)

    date_time = datetime.now().strftime("%Y-%m-%d_%H%M")
    output = os.path.join('./checkpoint', date_time)
    os.makedirs(output, exist_ok=True)

    args = easydict.EasyDict({
        "image_path": '../rsc/dirty_mnist/',
        "label_path": "../rsc/dirty_mnist_2nd_answer.csv",
        "output": output,
        "kfold": kfold,
        "epochs": epochs,
        "batch_size": 10,
        "lr": 1e-4,
        "device": device,
    })

    mnist_transforms = {
        'train': A.Compose([
            A.Rotate(limit=50),
            A.OneOf([
                A.GridDistortion(
                    distort_limit=(-0.3, 0.3), border_mode=cv2.BORDER_CONSTANT, p=1),
                A.ShiftScaleRotate(
                    rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, p=1),
                A.ElasticTransform(
                    alpha_affine=10, border_mode=cv2.BORDER_CONSTANT, p=1),
            ], p=1),
            A.Cutout(
                num_holes=24, max_h_size=10, max_w_size=10, fill_value=0),
            AP.ToTensorV2(),
        ]),
        'valid': A.Compose([
            AP.ToTensorV2(),
        ])
    }

    assert os.path.isdir(args.image_path), 'wrong path'
    assert os.path.isfile(args.label_path), 'wrong path'
    assert args.kfold < 5

    with open('../idx.pkl', 'rb') as f:
        arr = pickle.load(f)
    size = 50000
    split_gap = int(size * 0.2)
    split_init = args.kfold * split_gap
    split_end = split_init + split_gap

    valid_idx = arr[split_init:split_end]

    data_set = pd.read_csv(args.label_path)
    train_data = data_set.drop(valid_idx)
    valid_data = data_set.iloc[valid_idx]

    train_set = DatasetMNIST(
        image_folder=args.image_path,
        label_df=train_data,
        transforms=mnist_transforms['train']
    )

    valid_set = DatasetMNIST(
        image_folder=args.image_path,
        label_df=valid_data,
        transforms=mnist_transforms['valid']
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=args.batch_size,
        shuffle=False,
    )

    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    criterion = torch.nn.MultiLabelSoftMarginLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode='min',
        patience=2,
        factor=0.5,
        verbose=True
    )

    train_loss = []
    train_acc = []
    valid_loss = []
    valid_acc = []

    best_loss = float("inf")

    SAVE_DIR = args.output

    print('[info msg] training start !!\n')
    startTime = datetime.now()
    for epoch in range(1, args.epochs+1):
        print('Epoch {}/{}'.format(epoch, args.epochs))
        train_epoch_loss, train_epoch_acc = train(
            train_data_loader,
            model,
            criterion,
            args.device,
            optimizer
        )
        train_loss.append(train_epoch_loss)
        train_acc.append(train_epoch_acc)

        valid_epoch_loss, valid_epoch_acc = validate(
            valid_data_loader,
            model,
            criterion,
            args.device,
            scheduler
        )
        valid_loss.append(valid_epoch_loss)
        valid_acc.append(valid_epoch_acc)

        if best_loss > valid_epoch_loss:
            best_loss = valid_epoch_loss

            Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(
                SAVE_DIR, 'model_best_{}.pth.tar'.format(args.kfold)))
            print('Best model is saved in {}!!!'.format(date_time))

        else:
            torch.save(model.state_dict(), os.path.join(
                SAVE_DIR, 'model_last_{}.pth.tar'.format(args.kfold)))

        # save model weight at each 5 epochs
        if epoch % 5 == 0:
            date_time = datetime.now().strftime("%m-%d_%H%M")
            torch.save(model.state_dict(), os.path.join(
                SAVE_DIR, 'model_{}_{}_{}.pth.tar'.format(epoch, args.kfold, date_time)))

    elapsed_time = datetime.now() - startTime

    train_loss = np.array(train_loss)
    train_acc = np.array(train_acc)
    valid_loss = np.array(valid_loss)
    valid_acc = np.array(valid_acc)

    best_loss_pos = np.argmin(valid_loss)

    print('=' * 50)
    print('[info msg] training is done\n')
    print("Time taken: {}".format(elapsed_time))
    print("best loss is {} w/ acc {} at epoch : {}".format(best_loss,
          valid_acc[best_loss_pos], best_loss_pos))

    print('=' * 50)
    print('[info msg] {} model weight and log is save to {}\n'.format(
        args.model, SAVE_DIR))


class Network(nn.Module):
    def __init__(self, init=False):
        super().__init__()
        self.conv2d = nn.Conv2d(1, 3, 3, stride=1)
        self.regnet = regnet.regnetx_064()
        self.FC = nn.Linear(1000, 26)

        if init:
            model_weight = 'RegNetX-6.4G-5f725d05.pth'
            self.regnet.load_state_dict(torch.load(model_weight))

    def forward(self, x):
        x = F.relu(self.conv2d(x))
        return self.FC(self.regnet(x))


class DatasetMNIST(torch.utils.data.Dataset):
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


def train(train_loader, model, loss_func, device, optimizer, scheduler=None):
    n = 0
    running_loss = 0.0
    running_corrects = 0

    epoch_loss = 0.0
    epoch_acc = 0.0

    model.train()

    with tqdm.tqdm(train_loader, total=len(train_loader), desc="Train", file=sys.stdout) as iterator:
        for train_x, train_y in iterator:
            train_x = train_x.float().to(device)
            train_y = train_y.float().to(device)

            output = model(train_x)

            loss = loss_func(output, train_y)

            n += train_x.size(0)
            running_loss += loss.item() * train_x.size(0)

            epoch_loss = running_loss / float(n)

            output = output > 0.5
            # ----------------------------------------------------------------
            running_corrects += (output == train_y).cpu().sum().item()
            epoch_acc = running_corrects / train_y.size(1) / n

            log = 'loss - {:.5f}, acc - {:.5f}'.format(epoch_loss, epoch_acc)

            iterator.set_postfix_str(log)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            xm.mark_step()
            # ----------------------------------------------------------------

    if scheduler:
        scheduler.step(epoch_loss)

    return epoch_loss, epoch_acc


def validate(valid_loader, model, loss_func, device, scheduler=None):
    n = 0
    running_loss = 0.0
    running_corrects = 0

    epoch_loss = 0.0
    epoch_acc = 0.0

    model.eval()

    with tqdm.tqdm(valid_loader, total=len(valid_loader), desc="Valid", file=sys.stdout) as iterator:
        for train_x, train_y in iterator:
            train_x = train_x.float().to(device)
            train_y = train_y.float().to(device)

            with torch.no_grad():
                output = model(train_x)

            loss = loss_func(output, train_y)

            n += train_x.size(0)
            running_loss += loss.item() * train_x.size(0)

            epoch_loss = running_loss / float(n)

            output = output > 0.42
            # ----------------------------------------------------------------
            running_corrects += (output == train_y).cpu().sum().item()
            epoch_acc = running_corrects / train_y.size(1) / n

            log = 'loss - {:.5f}, acc - {:.5f}'.format(epoch_loss, epoch_acc)

            iterator.set_postfix_str(log)

    return epoch_loss, epoch_acc


if __name__ == '__main__':
    main()
