import argparse
import os
from pathlib import Path

import pandas as pd
import timm
import torch
import torch.backends.cudnn as cudnn
import wandb
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from dataset import create_transforms, MCDS
from utils import set_seed, testing_loop, training_loop, validation_loop

LR_DROP = 1000.
ROOT = Path('data/')


def main(config: argparse.ArgumentParser) -> None:

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu

    SAVE_PATH = Path(config.save_dir)
    if not os.path.isdir(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = False

    set_seed()

    train_tfms, val_tfms = create_transforms(config.imsize)

    df = pd.read_csv(ROOT/config.df_path)
    train_df = df[df.fold != config.fold]
    val_df = df[df.fold == config.fold]

    train_ds = MCDS(train_df, transform=train_tfms)
    val_ds = MCDS(val_df, transform=val_tfms)

    batch_size = config.bs
    train_sampler = None

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=config.workers, pin_memory=True, sampler=train_sampler)

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=config.workers, pin_memory=True)

    title = f'timber_{config.arch}_{config.imsize}'
    extra = ''
    if config.use_fmix:
        extra += "_fmix"

    criterion = nn.CrossEntropyLoss()

    if config.arch == "eff_s":
        model = timm.create_model('tf_efficientnetv2_s_in21ft1k', pretrained=True)
        FEATURE_NUM = 1280
        model.classifier = torch.nn.Linear(in_features=FEATURE_NUM, out_features=config.num_classes, bias=True)

    model = model.cuda()

    if config.predict:
        testing_loop(val_tfms, model, config, f'{config.result_name}.csv', SAVE_PATH)
        return 1

    if config.validate:
        validation_loop(val_tfms, model, config, criterion, SAVE_PATH)
        return 1

    optimizer = AdamW(model.parameters(), weight_decay=config.wd)

    steps = len(train_loader) * config.ep
    lr_scheduler = OneCycleLR(optimizer, config.lr, total_steps=steps, final_div_factor=LR_DROP)

    filename = '{}_{}_lr{}_f{}{}.pth.tar'.format(title, config.ep, config.lr, config.fold, extra)

    if config.wandb:
        with wandb.init(project=config.wandb_project, entity=config.wandb_account, config=config):
            training_loop(train_loader, val_loader, model, config, criterion, optimizer, lr_scheduler,
                          filename, SAVE_PATH, use_wandb=config.wandb)
    else:
        training_loop(train_loader, val_loader, model, config, criterion, optimizer, lr_scheduler,
                      filename, SAVE_PATH, use_wandb=config.wandb)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--arch', type=str, default="eff_s")
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--df-path', type=str, default="train.csv")
    parser.add_argument('--ep', type=int, default=1)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--imsize', type=int, default=384)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num-classes', type=int, default=3)
    parser.add_argument('--save-dir', type=str, default="models")
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--test-fldr', type=str, default="test")
    parser.add_argument('--use-amp', action='store_true')
    parser.add_argument('--use-fmix', action='store_true')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb-account', type=str, default="")
    parser.add_argument('--wandb-project', type=str, default="timber")
    parser.add_argument('--wd', type=float, default=3e-4)
    parser.add_argument('--workers', type=int, default=4)

    config = parser.parse_args()
    main(config)
