def config_filename():
    """[summary]

    Returns:
        [type]: [description]
    """
    return 'config.yml'


def experiment_subdir(exp_id):
    """[summary]

    Args:
        exp_id ([type]): [description]

    Returns:
        [type]: [description]
    """
    assert 0 <= exp_id <= 9999
    return f'exp_{exp_id:04d}'


def git_filename():
    """[summary]

    Returns:
        [type]: [description]
    """
    return 'git.json'


def weight_best_filename():
    """[summary]

    Returns:
        [type]: [description]
    """
    return 'model_best.pth'


def train_list_filename(split_id):
    """[summary]

    Args:
        split_id ([type]): [description]

    Returns:
        [type]: [description]
    """
    return f'train_{split_id}.json'


def val_list_filename(split_id):
    """[summary]

    Args:
        split_id ([type]): [description]

    Returns:
        [type]: [description]
    """
    return f'val_{split_id}.json'


def dump_git_info(path):
    """[summary]

    Args:
        path ([type]): [description]
    """
    import json
    import git

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    git_info = {'version': '0.0.0', 'sha': sha}

    with open(path, 'w') as f:
        json.dump(git_info,
                  f,
                  ensure_ascii=False,
                  indent=4,
                  sort_keys=False,
                  separators=(',', ': '))


def get_aoi_from_path(path):
    """[summary]

    Args:
        path ([type]): [description]
    """
    # path: /data/spacenet7/spacenet7/{train_or_test}/{aoi}/images_masked/{filename}.tif
    import os.path
    return os.path.basename(os.path.dirname(os.path.dirname(path)))


def crop_center(array, crop_wh):
    """[summary]

    Args:
        array ([type]): [description]
        crop_wh ([type]): [description]

    Returns:
        [type]: [description]
    """
    _, h, w = array.shape
    crop_w, crop_h = crop_wh
    assert w >= crop_w
    assert h >= crop_h

    left = (w - crop_w) // 2
    right = crop_w + left
    top = (h - crop_h) // 2
    bottom = crop_h + top

    return array[:, top:bottom, left:right]


def dump_prediction_to_png(path, pred):
    """[summary]

    Args:
        path ([type]): [description]
        pred ([type]): [description]
    """
    import numpy as np
    from skimage import io

    c, h, w = pred.shape
    assert c <= 3

    array = np.zeros(shape=[h, w, 3], dtype=np.uint8)
    array[:, :, :c] = (pred * 255).astype(np.uint8).transpose((1, 2, 0))
    io.imsave(path, array)


def load_prediction_from_png(path, n_channels):
    """[summary]

    Args:
        path ([type]): [description]
        n_channels ([type]): [description]

    Returns:
        [type]: [description]
    """
    from skimage import io

    assert n_channels <= 3

    array = io.imread(path)
    pred = (array.astype(float) / 255.0)[:, :, :n_channels]
    return pred.transpose((2, 0, 1))  # HWC to CHW
