import os
import pickle
import random
import sys
from datetime import datetime
from pathlib import Path

import click
import cv2
import easydict
import numpy as np
import pandas as pd
import torch
import torch_xla.core.xla_model as xm
import tqdm
from efficientnet_pytorch import EfficientNet

from util.utils import DatasetMNIST, get_aug, train, validate


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
    Train efficient-b8 model, using TPU in gcp.

    input path : ../rsc

    checkpoint path : ./checkpoint

    model : "efficientnet b8"

    Ex>  

    \t
    python3 train.py --kfold 0

    \t
    python3 train.py --network checkpoint/2022-09-25_0614/model_best.pth --kfold 0
    """

    date_time = datetime.now().strftime("%Y-%m-%d_%H%M")
    output = os.path.join('./checkpoint', date_time)
    os.makedirs(output, exist_ok=True)

    device = xm.xla_device()
    args = easydict.EasyDict({
        "image_path": '../rsc/dirty_mnist/',
        "label_path": "../rsc/dirty_mnist_2nd_answer.csv",
        "output": output,
        "kfold": kfold,
        "epochs": epochs,
        "batch_size": 10,
        "lr": 1e-4,
        "patience": 12,
        "resume": network,
        "device": device
    })

    mnist_transforms = get_aug()

    assert os.path.isdir(args.image_path), 'wrong path'
    assert os.path.isfile(args.label_path), 'wrong path'
    if (args.resume):
        assert os.path.isfile(args.resume), 'wrong path'
    assert args.kfold < 5

    # for cross-validation
    # loading randomly created array
    # to fix the indices
    with open('../idx.pkl', 'rb') as f:
        arr = pickle.load(f)

    data_set = pd.read_csv(args.label_path)
    size = len(data_set)
    split_gap = int(size * 0.2)
    split_init = args.kfold * split_gap
    split_end = split_init + split_gap

    valid_idx = arr[split_init:split_end]

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

    # load efficientnet-b8
    # weights pretrained with ImageNet
    # or what you prepare for
    if (args.resume):
        model = EfficientNet.from_name(
            "efficientnet-b8", in_channels=1, num_classes=26, dropout_rate=0.5)
        model.load_state_dict(torch.load(args.resume))
        print('[info msg] pre-trained weight is loaded !!\n')
        print(args.resume)
        print('=' * 50)

    else:
        print('[info msg] model is created\n')
        model = EfficientNet.from_pretrained(
            "efficientnet-b8", in_channels=1, num_classes=26, dropout_rate=0.5, advprop=True)
        print('=' * 50)

    model = model.to(args.device)

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

    best_loss = np.Inf
    SAVE_DIR = args.output
    
    print('[info msg] training start !!\n')
    startTime = datetime.now()
    for epoch in range(1, args.epochs+1):
        print('Epoch {}/{}'.format(epoch, args.epochs))
        train_epoch_loss, train_epoch_acc = train(
            train_data_loader,
            model,
            criterion,
            args,
            optimizer,
            scheduler
        )
        train_loss.append(train_epoch_loss)
        train_acc.append(train_epoch_acc)

        valid_epoch_loss, valid_epoch_acc = validate(
            valid_data_loader,
            model,
            criterion,
            args
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
    print('[info msg] model weight and log is save to {}\n'.format(SAVE_DIR))

    with open(os.path.join(SAVE_DIR, 'log.txt'), 'w') as f:
        for key, value in vars(args).items():
            f.write('{} : {}\n'.format(key, value))

        f.write('\n')
        f.write('total ecpochs : {}\n'.format(str(train_loss.shape[0])))
        f.write('time taken : {}\n'.format(str(elapsed_time)))
        f.write('best_train_loss {} w/ acc {} at epoch : {}\n'.format(np.min(train_loss),
                train_acc[np.argmin(train_loss)], np.argmin(train_loss)))
        f.write('best_valid_loss {} w/ acc {} at epoch : {}\n'.format(np.min(valid_loss),
                valid_acc[np.argmin(valid_loss)], np.argmin(valid_loss)))


if __name__ == '__main__':
    main()
