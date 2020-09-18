import json
import os.path
from glob import glob

import torch.utils.data

from ..transforms import get_augmentation, get_preprocess
from ..utils import get_image_paths, train_list_filename, val_list_filename
from .spacenet7 import SpaceNet7Dataset, SpaceNet7TestDataset


def get_dataloader(config, is_train):
    """[summary]

    Args:
        config ([type]): [description]
        is_train (bool): [description]

    Returns:
        [type]: [description]
    """
    # get path to train/val json files
    split_id = config.INPUT.TRAIN_VAL_SPLIT_ID
    train_list = os.path.join(config.INPUT.TRAIN_VAL_SPLIT_DIR,
                              train_list_filename(split_id))
    val_list = os.path.join(config.INPUT.TRAIN_VAL_SPLIT_DIR,
                            val_list_filename(split_id))

    preprocessing = get_preprocess(config, is_test=False)
    augmentation = get_augmentation(config, is_train=is_train)

    if is_train:
        data_list_path = train_list
        batch_size = config.DATALOADER.TRAIN_BATCH_SIZE
        num_workers = config.DATALOADER.TRAIN_NUM_WORKERS
        shuffle = config.DATALOADER.TRAIN_SHUFFLE
    else:
        data_list_path = val_list
        batch_size = config.DATALOADER.VAL_BATCH_SIZE
        num_workers = config.DATALOADER.VAL_NUM_WORKERS
        shuffle = False

    dataset = SpaceNet7Dataset(config,
                               data_list_path,
                               augmentation=augmentation,
                               preprocessing=preprocessing)

    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       num_workers=num_workers)


def get_test_dataloader(config):
    """[summary]

    Args:
        config ([type]): [description]

    Returns:
        [type]: [description]
    """
    preprocessing = get_preprocess(config, is_test=True)
    augmentation = get_augmentation(config, is_train=False)

    # get full paths to image files
    if config.TEST_TO_VAL:
        # use val split for test.
        split_id = config.INPUT.TRAIN_VAL_SPLIT_ID
        val_list_path = os.path.join(config.INPUT.TRAIN_VAL_SPLIT_DIR,
                                     val_list_filename(split_id))

        with open(val_list_path) as f:
            val_list = json.load(f)
        image_paths = [data['image_masked'] for data in val_list]

    else:
        # use test data for test (default).
        test_dir = config.INPUT.TEST_DIR
        image_paths = get_image_paths(test_dir)

    dataset = SpaceNet7TestDataset(config,
                                   image_paths,
                                   augmentation=augmentation,
                                   preprocessing=preprocessing)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config.DATALOADER.TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATALOADER.TEST_NUM_WORKERS,
    )
