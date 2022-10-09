import copy
import os
import easydict
import sys
from datetime import datetime

import random
import albumentations
import albumentations.pytorch
import click
import cv2
import numpy as np
import pandas as pd
import regnet_pytorch.regnet as regnet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import ttach as tta
from tqdm import tqdm


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
        image_fn = os.path.join(self.image_folder, str(self.label_df.iloc[index, 0]).zfill(5) + '.png')

        image = cv2.imread(image_fn, cv2.IMREAD_GRAYSCALE)
        image = image.reshape([256, 256, 1])

        label = self.label_df.iloc[index, 1:].values.astype('float')

        if self.transforms:
            image = self.transforms(image=image)['image'] / 255.0

        return image, label
    

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@click.command()
@click.pass_context
@click.option('--network', 'network', help='Pretrained network pickle path', required=True)
@click.option('--input', 'input', default='../rsc/test_dirty_mnist/', help='Directory of test imgs')
@click.option('--sample', 'sample', default='../rsc/sample_submission.csv', help='Sample Dataframe')
@click.option('--output', 'output', default='./infer/', help='Directory of output dataframe going to be saved')
@click.option('--seed', 'seed', default=None, type=int, help='Random seed')
def main(
    ctx: click.Context,
    network: str,
    input: str,
    sample: str,
    output: str,
    seed: int
):
    """
    using pretrained network, let's infer test data.\n

    output dataframe has probability of multi label, so need to get together all fold in one dataframe and apply threshold to binary-classify.

    input: ../rsc/test_dirty_mnist/\n
    output: ./infer/

    if you want perfectly recover the result, make sure to give it seed.
    if output directory exits, it will be removed and recreated. 

    Ex> 

    \t
    python3 infer.py --network model_best.pth.tar --seed 2022

    \t
    python3 infer.py --network model_best.pth.tar --output ./infer --seed 2022

    """
    if seed is not None:
        seed_everything(seed)

    device = xm.xla_device()
    date_time = datetime.now().strftime("%Y-%m-%d_%H%M")
    output = os.path.join(output, date_time)
    os.makedirs(output, exist_ok=True)

    # 학습이 완료된 모델의 가중치를 yo folder에 넣습니다.
    # 5fold이므로 5개의 모델 weight가 들어가야합니다.
    args = easydict.EasyDict({
        "image_path": input,
        "label_path": sample,
        "out_path": output,
        "weight_path": network,
        "batch_size": 10,
        "device": device,
    })

    assert os.path.isdir(args.image_path), 'wrong input path'
    assert os.path.isfile(args.label_path), 'wrong sample path'
    assert os.path.isfile(args.weight_path), 'wrong network path'

    print('=' * 50)
    print('[info msg] arguments')
    for key, value in vars(args).items():
        print(key, ":", value)

    base_transforms = {
        'test': albumentations.Compose([
            albumentations.pytorch.ToTensorV2(),
        ]),
    }

    tta_transforms = tta.Compose([tta.Rotate90(angles=[0, 90, 180, 270])])

    test_df = pd.read_csv(args.label_path)
    test_set = DatasetMNIST(
        image_folder=args.image_path,
        label_df=test_df,
        transforms=base_transforms['test']
    )

    submission_df = copy.copy(test_df)

    model = Network()
    model.load_state_dict(torch.load(args.weight_path))
    model = model.to(device)
    print('=' * 50)
    print('[info msg] weight is loaded')

    test_data_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model.eval()
    tta_model = tta.ClassificationTTAWrapper(model, tta_transforms)

    batch_size = args.batch_size
    batch_index = 0

    print('=' * 50)
    print('[info msg] inference start')

    for i, (images, _) in enumerate(tqdm(test_data_loader)):
        images = images.to(args.device)
        outputs = tta_model(images).detach().cpu().numpy().squeeze()
        batch_index = i * batch_size
        submission_df.iloc[batch_index:batch_index +
                            batch_size, 1:] += outputs

    SAVE_FN = args.out_path + '/result.csv'

    submission_df.to_csv(
        SAVE_FN,
        index=False
)

    print('=' * 50)
    print('[info msg] submission fils is saved to {}'.format(SAVE_FN))


if __name__ == '__main__':
    main()
