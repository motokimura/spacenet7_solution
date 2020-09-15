import os.path

import torch.utils.data

from ..transforms import get_augmentation, get_preprocess
from ..utils import train_list_filename, val_list_filename
from .spacenet7 import SpaceNet7Dataset


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
