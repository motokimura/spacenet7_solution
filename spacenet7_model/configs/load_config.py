import argparse
import os.path

from ..utils import config_filename, experiment_subdir, weight_best_filename
from .defaults import get_default_config


def _load_previous_experiment(config, exp_id, exp_log_dir, model_weight_dir):
    """[summary]

    Args:
        config ([type]): [description]
        exp_id ([type]): [description]
        exp_log_dir ([type]): [description]
        model_weight_dir ([type]): [description]

    Returns:
        [type]: [description]
    """
    exp_log_dir = exp_log_dir if exp_log_dir else config.LOG_ROOT
    model_weight_dir = model_weight_dir if model_weight_dir else config.WEIGHT_ROOT
    exp_subdir = experiment_subdir(exp_id)

    # overwrite config
    config_path = os.path.join(exp_log_dir, exp_subdir, config_filename())
    config.merge_from_file(config_path)

    # overwrite weight path
    weight_path = os.path.join(model_weight_dir, exp_subdir,
                               weight_best_filename())
    config.MODEL.WEIGHT = weight_path

    return config


def get_config_with_previous_experiment(exp_id=-1,
                                        exp_log_dir=None,
                                        model_weight_dir=None):
    """[summary]

    Args:
        exp_id (int, optional): [description]. Defaults to -1.
        exp_log_dir ([type], optional): [description]. Defaults to None.
        model_weight_dir ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    config = get_default_config()
    if exp_id >= 0:
        # overwrite hyper parameters with the ones used by previous experiment
        config = _load_previous_experiment(config, exp_id, exp_log_dir,
                                           model_weight_dir)
    else:
        # if exp_id < 0, return default config
        pass
    return config


def load_config():
    """[summary]

    Returns:
        [type]: [description]
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='path to YAML config file', type=str)
    parser.add_argument('--exp_id',
                        help='id of the previous experiment to load',
                        type=int,
                        default=-1)
    parser.add_argument(
        '--exp_log_dir',
        help='directory in which the previous experiment config is saved',
        type=str)
    parser.add_argument(
        '--model_weight_dir',
        help='directory in which the previous experiment model weight is saved',
        type=str)
    parser.add_argument('opts',
                        default=None,
                        help='parameter name and value pairs',
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = get_config_with_previous_experiment(
        exp_id=args.exp_id,
        exp_log_dir=args.exp_log_dir,
        model_weight_dir=args.model_weight_dir)
    if args.config:
        # overwrite hyper parameters with the ones given by the YAML file.
        config.merge_from_file(args.config)
    if args.opts:
        # overwrite hyper parameters with the ones given by command line args.
        config.merge_from_list(args.opts)
    config.freeze()

    return config
