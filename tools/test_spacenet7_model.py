#!/usr/bin/env python3
import os.path
import timeit

import _init_path
from spacenet7_model.configs import load_config
from spacenet7_model.datasets import get_test_dataloader
from spacenet7_model.models import get_model
from spacenet7_model.utils import (crop_center, dump_prediction_to_png,
                                   experiment_subdir, get_aoi_from_path)
from tqdm import tqdm


def main():
    """[summary]
    """
    config = load_config()
    print('successfully loaded config:')
    print(config)

    # prepare dataloader
    test_dataloader = get_test_dataloader(config)

    # prepare model to test
    model = get_model(config)
    model.eval()

    # prepare directory to output predictions
    exp_subdir = experiment_subdir(config.EXP_ID)
    pred_root = os.path.join(config.PREDICTION_ROOT, exp_subdir)
    os.makedirs(pred_root, exist_ok=False)

    # test loop
    for batch in tqdm(test_dataloader):
        images = batch['image'].to(config.MODEL.DEVICE)
        image_paths = batch['image_path']
        original_heights, original_widths, _ = batch['original_shape']

        predictions = model.module.predict(images)
        predictions = predictions.cpu().numpy()

        for i in range(len(predictions)):
            pred = predictions[i]
            path = image_paths[i]
            orig_h = original_heights[i].item()
            orig_w = original_widths[i].item()

            # prepare sub-directory under pred_root
            aoi = get_aoi_from_path(path)
            out_dir = os.path.join(pred_root, aoi)
            os.makedirs(out_dir, exist_ok=True)

            # remove padded area
            pred = crop_center(pred, crop_wh=(orig_w, orig_h))

            # dump to .png file
            filename = os.path.basename(path)
            filename, _ = os.path.splitext(filename)
            filename = f'{filename}.png'
            dump_prediction_to_png(os.path.join(out_dir, filename), pred)


if __name__ == '__main__':
    t0 = timeit.default_timer()

    main()

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60.0))
