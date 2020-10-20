#!/usr/bin/env python3

import json
import os.path
import timeit

import numpy as np

import _init_path
from skimage import io
from spacenet7_model.configs import load_config
from spacenet7_model.utils import (dump_prediction_to_png, ensemble_subdir,
                                   experiment_subdir, get_aoi_from_path,
                                   get_image_paths, load_prediction_from_png,
                                   val_list_filename)
from tqdm import tqdm

if __name__ == '__main__':
    t0 = timeit.default_timer()

    config = load_config()

    assert len(config.ENSEMBLE_EXP_IDS) >= 1
    N = len(config.ENSEMBLE_EXP_IDS)

    # prepare ensemble weights
    if len(config.ENSEMBLE_WEIGHTS) == 0:
        weights = np.ones(shape=(N))
    else:
        assert len(config.ENSEMBLE_WEIGHTS) == N
        weights = np.array(config.ENSEMBLE_WEIGHTS)
    weights = weights / weights.sum()

    # get full paths to image files
    if config.TEST_TO_VAL:
        # use val split for test.
        data_list_path = os.path.join(
            config.INPUT.TRAIN_VAL_SPLIT_DIR,
            val_list_filename(config.INPUT.TRAIN_VAL_SPLIT_ID))
        with open(data_list_path) as f:
            data_list = json.load(f)
        image_paths = [data['image_masked'] for data in data_list]
    else:
        # use test data for test (default).
        image_paths = get_image_paths(config.INPUT.TEST_DIR)

    subdir = ensemble_subdir(config.ENSEMBLE_EXP_IDS)
    out_root = os.path.join(config.ENSEMBLED_PREDICTION_ROOT, subdir)
    os.makedirs(out_root, exist_ok=False)

    for image_path in tqdm(image_paths):
        image_orig = io.imread(image_path)
        image = image_orig[:, :, :3]
        roi_mask = image_orig[:, :, 3] > 0

        h, w = roi_mask.shape
        ensembled_score = np.zeros(shape=[len(config.INPUT.CLASSES), h, w])

        aoi = get_aoi_from_path(image_path)

        image_filename = os.path.basename(image_path)
        array_filename, _ = os.path.splitext(image_filename)
        array_filename = f'{array_filename}.png'

        out_dir = os.path.join(out_root, aoi)
        os.makedirs(out_dir, exist_ok=True)

        for exp_id, weight in zip(config.ENSEMBLE_EXP_IDS, weights):
            exp_subdir = experiment_subdir(exp_id)
            score_array = load_prediction_from_png(
                os.path.join(config.PREDICTION_ROOT, exp_subdir, aoi,
                             array_filename),
                n_channels=len(config.INPUT.CLASSES))
            score_array[:, np.logical_not(roi_mask)] = 0
            assert score_array.min() >= 0 and score_array.max() <= 1
            ensembled_score += score_array * weight

        assert ensembled_score.min() >= 0 and ensembled_score.max() <= 1
        dump_prediction_to_png(os.path.join(out_dir, array_filename),
                               ensembled_score)

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60.0))
