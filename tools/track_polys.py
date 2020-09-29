#!/usr/bin/env python3

import os
import timeit
from multiprocessing import Pool

import _init_path
from spacenet7_model.configs import load_config
from spacenet7_model.utils import (convert_geojsons_to_csv, ensemble_subdir,
                                   get_subdirs, map_wrapper, solution_filename,
                                   track_footprint_identifiers)
from tqdm import tqdm

if __name__ == '__main__':
    t0 = timeit.default_timer()

    config = load_config()

    assert len(config.ENSEMBLE_EXP_IDS) >= 1

    subdir = ensemble_subdir(config.ENSEMBLE_EXP_IDS)
    input_root = os.path.join(config.POLY_ROOT, subdir)
    aois = get_subdirs(input_root)

    json_root = os.path.join(config.TRACKED_POLY_ROOT, subdir)
    os.makedirs(json_root, exist_ok=False)

    iou_field = 'iou_score'
    id_field = 'Id'
    reverse_order = False
    verbose = True
    super_verbose = False
    n_thread = 8

    # track footprint and save the results as geojson files
    # prepare args and output directories
    input_args = []
    for i, aoi in enumerate(aois):
        json_dir = os.path.join(json_root, aoi)
        os.makedirs(json_dir, exist_ok=False)

        input_dir = os.path.join(input_root, aoi)

        input_args.append([
            track_footprint_identifiers, input_dir, json_dir,
            config.TRACKING_MIN_IOU, iou_field, id_field, reverse_order,
            verbose, super_verbose
        ])

    # run multiprocessing
    pool = Pool(processes=n_thread)
    with tqdm(total=len(input_args)) as t:
        for _ in pool.imap_unordered(map_wrapper, input_args):
            t.update(1)

    # convert the geojson files into solution.csv to be submitted
    json_dirs = [
        os.path.join(json_root, aoi) for aoi in get_subdirs(json_root)
    ]

    if config.SOLUTION_OUTPUT_PATH and config.SOLUTION_OUTPUT_PATH != 'none':
        # only for deployment phase
        out_path = config.SOLUTION_OUTPUT_PATH
    else:
        out_dir = os.path.join(config.SOLUTION_CSV_ROOT, subdir)
        os.makedirs(out_dir, exist_ok=False)
        out_path = os.path.join(out_dir, solution_filename())

    convert_geojsons_to_csv(json_dirs, out_path, population='proposal')

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60.0))
