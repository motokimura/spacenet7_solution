import functools
import random

import numpy as np

import albumentations as albu


def get_spacenet7_augmentation(config, is_train):
    """[summary]

    Args:
        config ([type]): [description]
        is_train (bool): [description]

    Returns:
        [type]: [description]
    """
    if is_train:
        augmentation = [
            # random flip
            albu.HorizontalFlip(p=config.TRANSFORM.TRAIN_HORIZONTAL_FLIP_PROB),
            albu.VerticalFlip(p=config.TRANSFORM.TRAIN_VERTICAL_FLIP_PROB),
            # random rotate
            albu.ShiftScaleRotate(
                scale_limit=0.0,
                rotate_limit=config.TRANSFORM.TRAIN_RANDOM_ROTATE_DEG,
                shift_limit=0.0,
                p=config.TRANSFORM.TRAIN_RANDOM_ROTATE_PROB,
                border_mode=0),
            # random crop
            albu.RandomCrop(width=config.TRANSFORM.TRAIN_RANDOM_CROP_SIZE[0],
                            height=config.TRANSFORM.TRAIN_RANDOM_CROP_SIZE[1],
                            always_apply=True),
            albu.Lambda(image=functools.partial(
                _random_brightness,
                brightness_std=config.TRANSFORM.TRAIN_RANDOM_BRIGHTNESS_STD,
                p=config.TRANSFORM.TRAIN_RANDOM_BRIGHTNESS_PROB)),
        ]
    else:
        augmentation = [
            albu.PadIfNeeded(min_width=config.TRANSFORM.TEST_SIZE[0],
                             min_height=config.TRANSFORM.TEST_SIZE[1],
                             always_apply=True,
                             border_mode=0)
        ]
    return albu.Compose(augmentation)


def _random_brightness(image, brightness_std, p=1.0, **kwargs):
    """[summary]

    Args:
        image ([type]): [description]
        brightness_std ([type]): [description]
        p (float, optional): [description]. Defaults to 1.0.

    Returns:
        [type]: [description]
    """
    if brightness_std <= 0:
        return image

    if random.random() >= p:
        return image

    gauss = np.random.normal(0, brightness_std)
    brightness_noise = gauss * image
    noised = image + brightness_noise

    return noised
