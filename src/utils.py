import random
from pathlib import Path, PosixPath
from argparse import ArgumentParser
from typing import Union

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import wandb
from albumentations import Compose
from torch.cuda.amp import autocast, GradScaler
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from dataset import MCDS, Test_DS
from fmix import fmix
from progress.bar import Bar

ROOT = Path('data/')
RESIZED_FOLDER = 'square'

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seed(seed: int = 13) -> None:
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    # Set a fixed value for the hash seed


def sort_test_files(path: PosixPath) -> str:
    return (path.stem).zfill(3)


def test(test_loader: DataLoader, model: Module) -> torch.Tensor:
    """Calculate loss and top-1 classification accuracy one the validation set."""

    cudnn.benchmark = False
    cudnn.deterministic = True

    # switch to evaluate mode
    model.eval()

    pred_probs = []

    with torch.no_grad():
        bar = Bar('Processing', max=len(test_loader))

        for step, data in enumerate(test_loader):

            images = data.cuda(non_blocking=True)
            outputs = model(images)
            preds = torch.sigmoid(outputs.detach())
            pred_probs.append(preds.cpu())

            # plot progress
            bar.suffix = '({batch}/{size}) | Total: {total:} | ETA: {eta:}'.format(
                        batch=step + 1,
                        size=len(test_loader),
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        )
            bar.next()
    bar.finish()

    return torch.cat(pred_probs)


def testing_loop(val_tfms: Compose, model: Module, config: ArgumentParser, result_name: str, SAVE_PATH: PosixPath) -> None:
    model_paths = [str(mp) for mp in SAVE_PATH.iterdir()]
    ckpt = torch.load(model_paths[0], map_location='cpu')
    model.load_state_dict(ckpt["state_dict"], strict=False)
    test_file_list = sorted(list(Path(ROOT/RESIZED_FOLDER/config.test_fldr).iterdir()), key=sort_test_files)
    id_list = [int(fp.stem) for fp in test_file_list]
    test_dataset = Test_DS(test_file_list, transform=val_tfms)
    test_loader = DataLoader(test_dataset, batch_size=config.bs, shuffle=False,
                             num_workers=config.workers, pin_memory=True)
    if len(model_paths) > 1:
        total_probs = []
        probs = test(test_loader, model)
        total_probs.append(probs)
        for i in range(1, len(model_paths)):
            ckpt = torch.load(SAVE_PATH/model_paths[i], map_location='cpu')
            model.load_state_dict(ckpt["state_dict"], strict=False)

            probs = test(test_loader, model)
            total_probs.append(probs)
        total_probs = torch.stack(total_probs).mean(0)
    else:
        total_probs = test(test_loader, model)

    res = torch.argmax(total_probs, dim=1).numpy()
    df = pd.DataFrame(zip(id_list, res), columns=['id', 'class'])
    df.loc[df['class'] == 2, 'class'] = 3
    df.to_csv(ROOT/f'{result_name}.csv', index=False)


def train(train_loader: DataLoader, model: Module, config: ArgumentParser, criterion: Module, optimizer: Optimizer, 
          scheduler: _LRScheduler, use_fmix: bool = False) -> torch.Tensor:

    cudnn.benchmark = True
    cudnn.deterministic = False

    scaler = GradScaler()

    # switch to train mode
    model.train()
    optimizer.zero_grad()

    losses = AverageMeter()
    accs = AverageMeter()

    bar = Bar('Processing', max=len(train_loader))

    for step, data in enumerate(train_loader):

        images = data[0].cuda(non_blocking=True)
        targets = data[1].cuda(non_blocking=True)
        batch_size = images.size(0)
        # print(images.shape, targets.shape)

        if config.use_amp:
            with autocast():
                if use_fmix:
                    if batch_size == 1:
                        continue
                    loss, outputs = fmix(images, targets, model, criterion)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, targets)
        else:
            if use_fmix:
                if batch_size == 1:
                    continue
                loss, outputs = fmix(images, targets, model, criterion)
            else:
                outputs = model(images)
                loss = criterion(outputs, targets)

        # measure accuracy and record loss
        losses.update(loss.item(), batch_size)
        pred_probs = torch.sigmoid(outputs.detach())
        acc = (pred_probs.argmax(1) == targets).sum().item() / batch_size
        accs.update(acc, batch_size)

        if config.use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
        # compute gradient and do SGD step
        optimizer.step()
        optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()

        # plot progress
        bar.suffix = '({batch}/{size}) | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Accuracy: {acc: .4f}'.format(
                    batch=step + 1,
                    size=len(train_loader),
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    acc=accs.avg
                    )
        bar.next()
    bar.finish()
    return losses.avg, accs.avg


def training_loop(train_loader: DataLoader, val_loader: DataLoader, model: Module, config: ArgumentParser, criterion: Module, 
                  optimizer: Optimizer, scheduler: _LRScheduler, filename: str, SAVE_PATH: PosixPath, use_wandb: bool = False) -> None:
    for epoch in range(config.ep):

        print('\nEpoch: [{:d} | {:d}]'.format(epoch+1, config.ep))
        # train for one epoch
        train_loss, tacc = train(train_loader, model, config, criterion, optimizer, scheduler, use_fmix=config.use_fmix)

        # evaluate on validation set
        valid_loss, vacc = validate(val_loader, model, config, criterion)

        if use_wandb:
            # append logger file
            wandb.log({"t_loss": train_loss, "t_acc": tacc, "v_loss": valid_loss, "v_acc": vacc})

    model_state = {
        'accuracy': vacc,
        'config': config,
        'epoch': epoch+1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(model_state, SAVE_PATH/filename)


def validate(val_loader: DataLoader, model: Module, criterion: Module) -> torch.Tensor:
    """Calculate loss and top-1 classification accuracy one the validation set."""

    accs = AverageMeter()
    losses = AverageMeter()

    cudnn.benchmark = False
    cudnn.deterministic = True

    # switch to evaluate mode
    model.eval()

    preds = []
    targs = []

    with torch.no_grad():

        bar = Bar('Processing', max=len(val_loader))

        for step, data in enumerate(val_loader):

            images = data[0].cuda(non_blocking=True)
            batch_size = images.size(0)
            targets = data[1].cuda(non_blocking=True)

            outputs = model(images)

            loss = criterion(outputs, targets)
            # measure accuracy and record loss
            losses.update(loss.item(), batch_size)
            pred_probs = torch.sigmoid(outputs.detach())

            preds.append(pred_probs)
            targs.append(targets)
            acc = (pred_probs.argmax(1) == targets).sum().item() / batch_size
            accs.update(acc, batch_size)

            # plot progress
            bar.suffix = '({batch}/{size}) | Total: {total:} | ETA: {eta:} | Loss: {loss:.3f} | Accuracy: {acc: .4f}'.format(
                        batch=step + 1,
                        size=len(val_loader),
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        acc=accs.avg
                        )
            bar.next()
    bar.finish()
    return losses.avg, accs.avg


def validation_loop(val_tfms: Compose, model: Module, config: ArgumentParser, criterion: Module, SAVE_PATH: PosixPath) -> None:
    model_paths = [mp for mp in SAVE_PATH.iterdir()]
    fold = int(str(model_paths[0]).split('_f')[1][0])
    print(model_paths[0].stem)
    ckpt = torch.load(model_paths[0], map_location='cpu')
    model.load_state_dict(ckpt["state_dict"], strict=False)

    df = pd.read_csv(ROOT/config.df_path)
    val_df = df[df.fold == fold]
    val_ds = MCDS(val_df, transform=val_tfms)

    val_loader = DataLoader(
        val_ds, batch_size=config.bs, shuffle=False,
        num_workers=config.workers, pin_memory=True)

    if len(model_paths) > 1:
        _, _, = validate(val_loader, model, criterion)
        for i in range(1, len(model_paths)):
            fold = int(str(model_paths[i]).split('_f')[1][0])
            print(model_paths[i].stem)
            ckpt = torch.load(model_paths[i], map_location='cpu')
            model.load_state_dict(ckpt["state_dict"], strict=False)

            df = pd.read_csv(ROOT/config.df_path)
            val_df = df[df.fold == fold]
            val_ds = MCDS(val_df, transform=val_tfms)
            val_loader = DataLoader(
                val_ds, batch_size=config.bs, shuffle=False,
                num_workers=config.workers, pin_memory=True)

            _, _, = validate(val_loader, model, criterion)
    else:
        _, _, = validate(val_loader, model, criterion)
