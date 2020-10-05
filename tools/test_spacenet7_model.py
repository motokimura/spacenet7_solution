#!/usr/bin/env python3
import os.path
import timeit

import albumentations as albu
import cv2
import numpy as np
from tqdm import tqdm

import _init_path
from spacenet7_model.configs import load_config
from spacenet7_model.datasets import get_test_dataloader
from spacenet7_model.models import get_model
from spacenet7_model.utils import (crop_center, dump_prediction_to_png,
                                   experiment_subdir, get_aoi_from_path)


def main():
    """[summary]
    """
    config = load_config()
    print('successfully loaded config:')
    print(config)

    # prepare dataloaders
    test_dataloaders = []
    test_dataloaders.append(get_test_dataloader(config))  # default (w/o tta)
    for tta_width, tta_height in config.TTA.RESIZE:
        tta = albu.Resize(width=tta_width,
                          height=tta_height,
                          p=1.0,
                          always_apply=True)
        test_dataloaders.append(get_test_dataloader(config, tta=tta))
    N_dataloaders = len(test_dataloaders)

    # prepare model to test
    model = get_model(config)
    model.eval()

    # prepare directory to output predictions
    exp_subdir = experiment_subdir(config.EXP_ID)
    pred_root = os.path.join(config.PREDICTION_ROOT, exp_subdir)
    os.makedirs(pred_root, exist_ok=False)

    test_width, test_height = config.TRANSFORM.TEST_SIZE

    # test loop
    for batches in tqdm(zip(*test_dataloaders)):
        # prepare buffers for image file name and predicted array
        batch_size = len(batches[0]['image'])
        filenames = [None] * batch_size
        orig_image_sizes = [None] * batch_size
        predictions_averaged = np.zeros(shape=[
            batch_size,
            len(config.INPUT.CLASSES), test_height, test_width
        ])

        for dataloader_idx, batch in enumerate(batches):
            images = batch['image'].to(config.MODEL.DEVICE)
            image_paths = batch['image_path']
            original_heights, original_widths, _ = batch['original_shape']

            predictions = model.module.predict(images)
            predictions = predictions.cpu().numpy()

            for batch_idx in range(len(predictions)):
                pred = predictions[batch_idx]
                path = image_paths[batch_idx]
                orig_h = original_heights[batch_idx].item()
                orig_w = original_widths[batch_idx].item()

                # prepare sub-directory under pred_root
                aoi = get_aoi_from_path(path)
                out_dir = os.path.join(pred_root, aoi)
                os.makedirs(out_dir, exist_ok=True)

                # resize
                pred = pred.transpose(1, 2, 0)  # CHW -> HWC
                pred = cv2.resize(pred, dsize=(test_width, test_height))
                pred = pred.transpose(2, 0, 1)  # HWC -> CHW

                # store predictions into the buffer
                predictions_averaged[batch_idx] += pred / N_dataloaders

                # store image filenemes and sizes into the buffers
                filename = os.path.basename(path)
                orig_image_wh = (orig_w, orig_h)
                if dataloader_idx == 0:
                    filenames[batch_idx] = filename
                    orig_image_sizes[batch_idx] = orig_image_wh
                else:
                    assert filenames[batch_idx] == filename
                    assert orig_image_sizes[batch_idx] == orig_image_wh

        for filename, orig_image_wh, pred_averaged in zip(
                filenames, orig_image_sizes, predictions_averaged):
            # remove padded area
            pred_averaged = crop_center(pred_averaged, crop_wh=orig_image_wh)

            # dump to .png file
            filename, _ = os.path.splitext(filename)
            filename = f'{filename}.png'
            dump_prediction_to_png(os.path.join(out_dir, filename),
                                   pred_averaged)


if __name__ == '__main__':
    t0 = timeit.default_timer()

    main()

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60.0))
