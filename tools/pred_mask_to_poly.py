#!/usr/bin/env python3

import os
import timeit
from glob import glob

import _init_path
from spacenet7_model.configs import load_config
from spacenet7_model.utils import (ensemble_subdir, get_subdirs,
                                   load_prediction_from_png,
                                   compute_building_score,
                                   gen_building_polys_using_contours)
from tqdm import tqdm

if __name__ == '__main__':
    t0 = timeit.default_timer()

    config = load_config()

    assert len(config.ENSEMBLE_EXP_IDS) >= 1

    subdir = ensemble_subdir(config.ENSEMBLE_EXP_IDS)
    input_root = os.path.join(config.ENSEMBLED_PREDICTION_ROOT, subdir)
    aois = get_subdirs(input_root)

    out_root = os.path.join(config.POLY_ROOT, subdir)
    os.makedirs(out_root, exist_ok=False)

    for aoi in aois:
        paths = glob(os.path.join(input_root, aoi, '*.png'))
        paths.sort()

        out_dir = os.path.join(out_root, aoi)
        os.makedirs(out_dir, exist_ok=False)

        for path in tqdm(paths):
            pred_array = load_prediction_from_png(path,
                                                  n_channels=len(
                                                      config.INPUT.CLASSES))

            footprint_channel = config.INPUT.CLASSES.index(
                'building_footprint')
            boundary_channel = config.INPUT.CLASSES.index('building_boundary')
            contact_channel = config.INPUT.CLASSES.index('building_contact')

            footprint_score = pred_array[footprint_channel]
            boundary_score = pred_array[boundary_channel]
            contact_score = pred_array[contact_channel]
            building_score = compute_building_score(
                footprint_score,
                boundary_score,
                contact_score,
                alpha=config.BOUNDARY_SUBTRACT_COEFF,
                beta=config.CONTACT_SUBTRACT_COEFF)

            filename = os.path.basename(path)
            filename, _ = os.path.splitext(filename)
            output_path = os.path.join(out_dir, f'{filename}.geojson')

            if config.METHOD_TO_MAKE_POLYGONS == 'contours':
                polys = gen_building_polys_using_contours(
                    building_score,
                    config.BUILDING_MIM_AREA_PIXEL,
                    config.BUILDING_SCORE_THRESH,
                    simplify=False,
                    output_path=output_path)
            elif config.METHOD_TO_MAKE_POLYGONS == 'watershed':
                raise NotImplementedError()
            else:
                raise ValueError()

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60.0))
