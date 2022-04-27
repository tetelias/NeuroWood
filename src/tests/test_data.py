import os
import unittest
from pathlib import Path

from PIL import Image

from preprocessing import EXTRA_TEST_FLDR

FOLDERS = ['data/drova/', 'data/1/', 'data/3/', 'data/test/', 'data/test_top_scores/']
TEST_FLDR = 'data/square/test'
TRAIN_FLDR = 'data/square/train'
FOLDER_SIZES = [69, 299, 210, 249, 421]
VALIDATION_IMSIZE = 512


class TestDataQuality(unittest.TestCase):

    # Testing presence of 5 raw data subfolders
    def test_raw_data_present_class_drova(self):
        self.assertTrue(os.path.isdir(FOLDERS[0]))

    def test_raw_data_present_class_1(self):
        self.assertTrue(os.path.isdir(FOLDERS[1]))

    def test_raw_data_present_class_3(self):
        self.assertTrue(os.path.isdir(FOLDERS[2]))

    def test_raw_data_present_test(self):
        self.assertTrue(os.path.isdir(FOLDERS[3]))

    def test_raw_data_present_test_top_scores(self):
        self.assertTrue(os.path.isdir(FOLDERS[4]))

    # Testing size of 5 raw data subfolders
    def test_raw_data_non_empty_class_drova(self):
        self.assertEqual(len(os.listdir(FOLDERS[0])), FOLDER_SIZES[0])

    def test_raw_data_non_empty_class_1(self):
        self.assertEqual(len(os.listdir(FOLDERS[1])), FOLDER_SIZES[1])

    def test_raw_data_non_empty_class_3(self):
        self.assertEqual(len(os.listdir(FOLDERS[2])), FOLDER_SIZES[2])

    def test_raw_data_non_empty_test(self):
        self.assertEqual(len(os.listdir(FOLDERS[3])), FOLDER_SIZES[3])

    def test_raw_data_non_empty_test_top_scores(self):
        self.assertEqual(len(os.listdir(FOLDERS[4])), FOLDER_SIZES[4])

    # Testing validity of images in 5 raw data subfolders
    def test_raw_data_valid_images_class_drova(self):
        non_valid_count = 0
        for fp in Path(FOLDERS[0]).iterdir():
            try:
                with Image.open(fp) as img:
                    size = img.size
                    self.assertNotEqual(size[0], 0)
                    self.assertNotEqual(size[1], 0)
            except AttributeError:
                non_valid_count += 1
        self.assertEqual(non_valid_count, 0)

    def test_raw_data_valid_images_class_1(self):
        non_valid_count = 0
        for fp in Path(FOLDERS[1]).iterdir():
            try:
                with Image.open(fp) as img:
                    size = img.size
                    self.assertNotEqual(size[0], 0)
                    self.assertNotEqual(size[1], 0)
            except AttributeError:
                non_valid_count += 1
        self.assertEqual(non_valid_count, 0)

    def test_raw_data_valid_images_class_3(self):
        non_valid_count = 0
        for fp in Path(FOLDERS[2]).iterdir():
            try:
                with Image.open(fp) as img:
                    size = img.size
                    self.assertNotEqual(size[0], 0)
                    self.assertNotEqual(size[1], 0)
            except AttributeError:
                non_valid_count += 1
        self.assertEqual(non_valid_count, 0)

    def test_raw_data_valid_images_test(self):
        non_valid_count = 0
        for fp in Path(FOLDERS[3]).iterdir():
            try:
                with Image.open(fp) as img:
                    size = img.size
                    self.assertNotEqual(size[0], 0)
                    self.assertNotEqual(size[1], 0)
            except AttributeError:
                non_valid_count += 1
        self.assertEqual(non_valid_count, 0)

    def test_raw_data_valid_images_test_top_scores(self):
        non_valid_count = 0
        for fp in Path(FOLDERS[4]).iterdir():
            try:
                with Image.open(fp) as img:
                    size = img.size
                    self.assertNotEqual(size[0], 0)
                    self.assertNotEqual(size[1], 0)
            except AttributeError:
                non_valid_count += 1
        self.assertEqual(non_valid_count, 0)

    # Testing correct resizing of images
    def test_raw_data_valid_testing_resized_images(self):
        non_valid_count = 0
        for fp in Path(TEST_FLDR).iterdir():
            try:
                with Image.open(fp) as img:
                    size = img.size
                    self.assertEqual(size[0], VALIDATION_IMSIZE)
                    self.assertEqual(size[1], VALIDATION_IMSIZE)
            except AttributeError:
                non_valid_count += 1
        self.assertEqual(non_valid_count, 0)

    def test_raw_data_valid_testing_top_scores_resized_images(self):
        non_valid_count = 0
        for fp in Path(EXTRA_TEST_FLDR).iterdir():
            try:
                with Image.open(fp) as img:
                    size = img.size
                    self.assertEqual(size[0], VALIDATION_IMSIZE)
                    self.assertEqual(size[1], VALIDATION_IMSIZE)
            except AttributeError:
                non_valid_count += 1
        self.assertEqual(non_valid_count, 0)

    def test_raw_data_valid_training_resized_images(self):
        non_valid_count = 0
        for fp in Path(TRAIN_FLDR).iterdir():
            try:
                with Image.open(fp) as img:
                    size = img.size
                    self.assertEqual(size[0], VALIDATION_IMSIZE)
                    self.assertEqual(size[1], VALIDATION_IMSIZE)
            except AttributeError:
                non_valid_count += 1
        self.assertEqual(non_valid_count, 0)

    # # Testing the correct amount of resized images
    def test_raw_data_valid_testing_resized_images_count(self):
        valid_count = 0
        for fp in Path(TEST_FLDR).iterdir():
            try:
                with Image.open(fp) as img:
                    _ = img.size
                    valid_count += 1
            except AttributeError:
                pass
        self.assertEqual(valid_count, FOLDER_SIZES[3])

    def test_raw_data_valid_testing_top_scores_resized_images_count(self):
        valid_count = 0
        for fp in Path(EXTRA_TEST_FLDR).iterdir():
            try:
                with Image.open(fp) as img:
                    _ = img.size
                    valid_count += 1
            except AttributeError:
                pass
        self.assertEqual(valid_count, FOLDER_SIZES[4])

    def test_raw_data_valid_training_resized_images_count(self):
        valid_count = 0
        for fp in Path(TRAIN_FLDR).iterdir():
            try:
                with Image.open(fp) as img:
                    _ = img.size
                    valid_count += 1
            except AttributeError:
                pass
        self.assertEqual(valid_count, sum(FOLDER_SIZES[:3]))
