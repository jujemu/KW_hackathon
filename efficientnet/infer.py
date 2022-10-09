import os
from datetime import datetime

import click
import easydict
import pandas as pd
import torch
import torch_xla.core.xla_model as xm
import ttach as tta
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader

from util.utils import DatasetMNIST, get_aug, infer, seed_everything


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

    args = easydict.EasyDict({
        "input": input,
        "sample": sample,
        "output": output,
        "batch_size": 10,
        "network": network,
        "device": device
    })

    transforms = get_aug()
    sample = pd.read_csv(args.sample)
    ds = DatasetMNIST(args.input, sample, transforms['valid'])
    loader = DataLoader(ds, args.batch_size)

    tta_transforms = tta.Compose([tta.Rotate90(angles=[0, 90, 180, 270])])

    # inference
    result = sample
    model = EfficientNet.from_name("efficientnet-b8", in_channels=1, num_classes=26)
    model.load_state_dict(torch.load(args.network, map_location=torch.device('cpu')))
    print('=' * 50)
    print('[info msg] weight is loaded')
    
    model = model.to(device)
    model.eval()
    tta_model = tta.ClassificationTTAWrapper(model, tta_transforms)

    result = infer(loader, tta_model, args, result)
    result.to_csv(os.path.join(args.output, 'result.csv'), index=False)
    

if __name__ == "__main__":
    main()
