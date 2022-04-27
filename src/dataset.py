from typing import Union
import cv2
import torch
from pandas import DataFrame
from torch.utils.data import Dataset
from albumentations import Compose, GridDropout, HorizontalFlip, Normalize, PadIfNeeded, \
    RandomCrop, RandomRotate90, VerticalFlip
from albumentations.pytorch import ToTensorV2

NORM = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]


def create_transforms(imsize: int, test_imsize: int = 512) -> Compose:

    train_tfms = Compose([
        RandomCrop(imsize, imsize),
        HorizontalFlip(),
        VerticalFlip(),
        RandomRotate90(),
        GridDropout(unit_size_min=imsize//3, unit_size_max=imsize, random_offset=True, mask_fill_value=None, p=1.0),
        Normalize(
            mean=NORM[0],
            std=NORM[1],
        ),
        ToTensorV2(),
    ])

    val_tfms = Compose([
        PadIfNeeded(test_imsize, test_imsize),
        Normalize(
            mean=NORM[0],
            std=NORM[1],
        ),
        ToTensorV2()
    ])
    return train_tfms, val_tfms


class MCDS(Dataset):
    """Dataset for multi-label classification"""

    def __init__(self, df: DataFrame, transform: Compose = None) -> None:

        self.img_paths = df['path'].values
        self.transform = transform
        self.y = df['label'].values

    def __getitem__(self, index: int) -> torch.Tensor:
        # img = cv2.imread(str(self.img_paths[index]))[:, :, ::-1]
        img = cv2.imread(str(self.img_paths[index]), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label = self.y[index]

        if self.transform is not None:
            augmented = self.transform(image=img)
            img = augmented['image']

        return img, label

    def __len__(self) -> int:
        return len(self.y)


class Test_DS(Dataset):
    """Dataset for multi-label classification"""

    def __init__(self, path_list: list, transform: Compose = None) -> None:

        self.img_paths = path_list
        self.transform = transform

    def __getitem__(self, index) -> torch.Tensor:
        img = cv2.imread(str(self.img_paths[index]), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            augmented = self.transform(image=img)
            img = augmented['image']
        return img

    def __len__(self) -> int:
        return len(self.img_paths)
