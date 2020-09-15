import json

import git


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
