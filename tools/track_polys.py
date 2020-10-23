#!/usr/bin/env python3

import multiprocessing as mp
import os
import timeit

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

    # prepare json and output directories
    out_root = os.path.join(config.TRACKED_POLY_ROOT, subdir)
    os.makedirs(out_root, exist_ok=False)

    if config.SOLUTION_OUTPUT_PATH and config.SOLUTION_OUTPUT_PATH != 'none':
        # only for deployment phase
        out_path = config.SOLUTION_OUTPUT_PATH
    else:
        out_path = os.path.join(out_root, solution_filename())

    # some parameters
    iou_field = 'iou_score'
    id_field = 'Id'
    reverse_order = config.TRACKING_REVERSE
    num_next_frames = config.TRACKING_NUM_AHEAD_FRAMES
    min_iou_frames = config.TRACKING_MIN_IOU_NEW_BUILDING
    shape_update_method = config.TRACKING_SHAPE_UPDATE_METHOD
    max_area_occupied = config.TRACKING_MAX_AREA_OCCUPIED
    verbose = True
    super_verbose = False

    n_thread = config.TRACKING_NUM_THREADS
    n_thread = n_thread if n_thread > 0 else mp.cpu_count()
    print(f'N_thread for multiprocessing: {n_thread}')

    # track footprint and save the results as geojson files
    # prepare args and output directories
    input_args = []
    for i, aoi in enumerate(aois):
        json_dir = os.path.join(out_root, aoi)
        os.makedirs(json_dir, exist_ok=False)

        input_dir = os.path.join(input_root, aoi)

        input_args.append([
            track_footprint_identifiers, input_dir, json_dir,
            config.TRACKING_MIN_IOU, iou_field, id_field, reverse_order,
            num_next_frames, min_iou_frames, shape_update_method,
            max_area_occupied, verbose, super_verbose
        ])

    # run multiprocessing
    pool = mp.Pool(processes=n_thread)
    with tqdm(total=len(input_args)) as t:
        for _ in pool.imap_unordered(map_wrapper, input_args):
            t.update(1)

    # convert the geojson files into solution.csv to be submitted
    json_dirs = [os.path.join(out_root, aoi) for aoi in get_subdirs(out_root)]

    convert_geojsons_to_csv(json_dirs, out_path, population='proposal')

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60.0))
