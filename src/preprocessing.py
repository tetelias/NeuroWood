from pathlib import Path, PosixPath

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

DF_FILE = 'data/train.csv'
SOURCE_FOLDERS = ['data/drova/', 'data/1/', 'data/3/', 'data/test_top_scores/', 'data/test/']
EXTRA_TEST_FLDR = 'data/square/test_top_scores'
TEST_FLDR = 'data/square/test'
TRAIN_FLDR = 'data/square/train'
CLASSES_SIZE = [69, 299, 210]
IMG_SIZE = 512


def main() -> None:
    Path(EXTRA_TEST_FLDR).mkdir(parents=True, exist_ok=True)
    Path(TEST_FLDR).mkdir(parents=True, exist_ok=True)
    Path(TRAIN_FLDR).mkdir(parents=True, exist_ok=True)

    impaths = []
    for fldr in SOURCE_FOLDERS:
        if 'test/' in fldr:
            resize_folder(fldr, dest_fldr=TEST_FLDR)
        elif 'test_top_scores/' in fldr:
            resize_folder(fldr, dest_fldr=EXTRA_TEST_FLDR)
        else:
            resize_folder(fldr)
            impaths.extend(collect_paths(fldr))

    classes = [0] * CLASSES_SIZE[0] + [1] * CLASSES_SIZE[1] + [2] * CLASSES_SIZE[2]
    df = pd.DataFrame(zip(impaths, classes), columns=['path', 'label'])
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)
    fold = 0
    for _, test_index in kfold.split(df, df['label']):
        df.loc[test_index, 'fold'] = fold
        fold += 1
    df.fold = df.fold.astype('uint8')

    df.to_csv(DF_FILE, index=False)


def collect_paths(src_fldr: str) -> list:
    paths_list = []
    for fp in Path(src_fldr).iterdir():
        paths_list.append(f'{TRAIN_FLDR}/{fp.name}')
    return paths_list


def resize_folder(src_fldr: str, dest_fldr: str = TRAIN_FLDR) -> None:
    for fp in Path(src_fldr).iterdir():
        img = resize_img(fp)
        cv2.imwrite(str(Path(dest_fldr)/fp.name), img)


def resize_img(fpath: PosixPath) -> np.array:
    img = cv2.imread(str(fpath), cv2.IMREAD_UNCHANGED)
    h, w = img.shape[:2]
    if h >= w:
        img = cv2.resize(img[(h - w)//2:(w + h)//2], (IMG_SIZE, IMG_SIZE))
    elif h < w:
        img = cv2.resize(img[:, (w - h)//2:(w + h)//2], (IMG_SIZE, IMG_SIZE))
    return img


if __name__ == '__main__':
    main()
