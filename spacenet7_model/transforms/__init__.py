from .augmentations import get_spacenet7_augmentation
from .preprocesses import get_spacenet7_preprocess


def get_preprocess(config, is_test):
    """[summary]

    Args:
        config ([type]): [description]
        is_test (bool): [description]

    Returns:
        [type]: [description]
    """
    return get_spacenet7_preprocess(config, is_test)


def get_augmentation(config, is_train):
    """[summary]

    Args:
        config ([type]): [description]
        is_train (bool): [description]

    Returns:
        [type]: [description]
    """
    return get_spacenet7_augmentation(config, is_train)
