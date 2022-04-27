import os
import unittest

import pandas as pd
from torch.utils.data import DataLoader

from src.dataset import create_transforms, MCDS

FOLD_SERIES = 'fold'
FOLDERS = ['data/1/', 'data/3/', 'data/drova/', 'data/test_top_scores/', 'data/test/']
LABELS_PATH = 'data/train.csv'
TEST_FLDR = 'data/square/test'
TRAIN_FLDR = 'data/square/train'
BATCH_SIZE = 4
CHANNELS = 3
FOLD = 4
FOLDER_SIZES = [299, 210, 69, 421, 249]
FOLD4_SIZE = 115
NON_FOLD4_SIZE = 463
NUM_FOLDS = 5
TRAIN_IMSIZE = 384
VALIDATION_IMSIZE = 512
WORKERS = 4


class TestTraining(unittest.TestCase):

    # Testing presence of a file with training labels
    def test_labels_presence(self):
        self.assertTrue(os.path.isfile(LABELS_PATH))

    # Testing amount of training labels
    def test_folds_presence(self):
        df = pd.read_csv(LABELS_PATH)
        self.assertEqual(len(df[FOLD_SERIES]), sum(FOLDER_SIZES[:3]))

    # Testing fold count
    def test_fold_sizes(self):
        df = pd.read_csv(LABELS_PATH)
        self.assertEqual(df[FOLD_SERIES].nunique(), NUM_FOLDS)

    # Testing size of training dataframe for fold #4
    def test_train_dataframe_size(self):
        df = pd.read_csv(LABELS_PATH)
        train_df = df[df.fold != FOLD]
        self.assertEqual(len(train_df[FOLD_SERIES]), NON_FOLD4_SIZE)

    # Testing size of validation dataframe for fold #4
    def test_validation_dataframe_size(self):
        df = pd.read_csv(LABELS_PATH)
        validation_df = df[df.fold == FOLD]
        self.assertEqual(len(validation_df[FOLD_SERIES]), FOLD4_SIZE)

    # Testing size of a sample produced by training dataset for fold #4
    def test_train_dataset_item(self):
        train_tfms, _ = create_transforms(TRAIN_IMSIZE)
        df = pd.read_csv(LABELS_PATH)
        train_df = df[df.fold != FOLD]
        train_ds = MCDS(train_df, transform=train_tfms)
        self.assertEqual(next(iter(train_ds))[0].size(), tuple([CHANNELS, TRAIN_IMSIZE, TRAIN_IMSIZE]))

    # Testing size of a sample produced by validation dataset for fold #4
    def test_validation_dataset_item(self):
        _, validation_tfms = create_transforms(VALIDATION_IMSIZE)
        df = pd.read_csv(LABELS_PATH)
        validation_df = df[df.fold == FOLD]
        validation_ds = MCDS(validation_df, transform=validation_tfms)
        self.assertEqual(next(iter(validation_ds))[0].size(), tuple([CHANNELS, VALIDATION_IMSIZE, VALIDATION_IMSIZE]))

    # Testing size of training dataset for fold #4
    def test_train_dataset_size(self):
        train_tfms, _ = create_transforms(TRAIN_IMSIZE)
        df = pd.read_csv(LABELS_PATH)
        train_df = df[df.fold != FOLD]
        train_ds = MCDS(train_df, transform=train_tfms)
        self.assertEqual(len(train_ds), NON_FOLD4_SIZE)

    # Testing size of validation dataset for fold #4
    def test_validation_dataset_size(self):
        _, validation_tfms = create_transforms(VALIDATION_IMSIZE)
        df = pd.read_csv(LABELS_PATH)
        validation_df = df[df.fold == FOLD]
        validation_ds = MCDS(validation_df, transform=validation_tfms)
        self.assertEqual(len(validation_ds), FOLD4_SIZE)

    # Testing size of a sample produced by  training dataloader for fold #4
    def test_train_dataloader_item(self):
        train_tfms, _ = create_transforms(TRAIN_IMSIZE)
        train_sampler = None
        df = pd.read_csv(LABELS_PATH)
        train_df = df[df.fold != FOLD]
        train_ds = MCDS(train_df, transform=train_tfms)
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=(train_sampler is None),
            num_workers=WORKERS, pin_memory=True, sampler=train_sampler)
        self.assertEqual(next(iter(train_loader))[0].size(), tuple([BATCH_SIZE, CHANNELS, TRAIN_IMSIZE, TRAIN_IMSIZE]))

    # Testing size of training dataloader for fold #4
    def test_train_dataloader_size(self):
        train_tfms, _ = create_transforms(TRAIN_IMSIZE)
        train_sampler = None
        df = pd.read_csv(LABELS_PATH)
        train_df = df[df.fold != FOLD]
        train_ds = MCDS(train_df, transform=train_tfms)
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=(train_sampler is None),
            num_workers=WORKERS, pin_memory=True, sampler=train_sampler)
        self.assertEqual(len(train_loader), (NON_FOLD4_SIZE + BATCH_SIZE - 1) // BATCH_SIZE)

    def test_validation_dataloader_item(self):
        _, validation_tfms = create_transforms(VALIDATION_IMSIZE)
        df = pd.read_csv(LABELS_PATH)
        validation_df = df[df.fold == FOLD]
        validation_ds = MCDS(validation_df, transform=validation_tfms)
        validation_loader = DataLoader(
            validation_ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=WORKERS, pin_memory=True)
        self.assertEqual(next(iter(validation_loader))[0].size(), tuple([BATCH_SIZE, CHANNELS, VALIDATION_IMSIZE, VALIDATION_IMSIZE]))

    def test_validation_dataloader_size(self):
        _, validation_tfms = create_transforms(VALIDATION_IMSIZE)
        df = pd.read_csv(LABELS_PATH)
        validation_df = df[df.fold == FOLD]
        validation_ds = MCDS(validation_df, transform=validation_tfms)
        validation_loader = DataLoader(
            validation_ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=WORKERS, pin_memory=True)
        self.assertEqual(len(validation_loader), (FOLD4_SIZE + BATCH_SIZE - 1) // BATCH_SIZE)
