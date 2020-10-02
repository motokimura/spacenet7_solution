def map_wrapper(x):
    """[summary]

    Args:
        x ([type]): [description]

    Returns:
        [type]: [description]
    """
    return x[0](*(x[1:]))


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


def ensemble_subdir(exp_ids):
    """[summary]

    Args:
        exp_ids ([type]): [description]

    Returns:
        [type]: [description]
    """
    exp_ids_ = sorted(exp_ids)
    subdir = 'exp'
    for exp_id in exp_ids_:
        subdir += f'_{exp_id:04d}'
    return subdir


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


def solution_filename():
    """[summary]

    Returns:
        [type]: [description]
    """
    return 'solution.csv'


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


def get_subdirs(input_dir):
    """[summary]

    Args:
        input_dir ([type]): [description]

    Returns:
        [type]: [description]
    """
    import os.path

    subdirs = sorted([
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
    ])

    return subdirs


def get_image_paths(input_dir):
    """[summary]

    Args:
        input_dir ([type]): [description]

    Returns:
        [type]: [description]
    """
    import os.path
    from glob import glob

    aois = get_subdirs(input_dir)

    image_paths = []
    for aoi in aois:
        paths = glob(os.path.join(input_dir, aoi, 'images_masked/*.tif'))
        paths.sort()
        image_paths.extend(paths)

    return image_paths


def get_aoi_from_path(path):
    """[summary]

    Args:
        path ([type]): [description]
    """
    # path: /data/spacenet7/spacenet7/{train_or_test}/{aoi}/images_masked/{filename}
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


def compute_building_score(pr_score_footprint, pr_score_boundary,
                           pr_score_contact, alpha, beta):
    """[summary]

    Args:
        pr_score_footprint ([type]): [description]
        pr_score_boundary ([type]): [description]
        pr_score_contact ([type]): [description]
        alpha ([type]): [description]
        beta ([type]): [description]

    Returns:
        [type]: [description]
    """
    pr_score_building = pr_score_footprint
    pr_score_building *= (1.0 - alpha * pr_score_boundary)
    pr_score_building *= (1.0 - beta * pr_score_contact)
    return pr_score_building.clip(min=0.0, max=1.0)


def save_empty_geojson(path):
    """[summary]

    Args:
        path ([type]): [description]
    """
    import json
    empty_dict = {"type": "FeatureCollection", "features": []}
    with open(path, 'w') as f:
        json.dump(empty_dict, f)


def gen_building_polys_using_contours(building_score,
                                      min_area_pix,
                                      score_thresh,
                                      simplify=False,
                                      output_path=None):
    """[summary]

    Args:
        building_score ([type]): [description]
        min_area_pix ([type]): [description]
        score_thresh ([type]): [description]
        simplify (bool, optional): [description]. Defaults to False.
        output_path ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    import solaris as sol
    polygon_gdf = sol.vector.mask.mask_to_poly_geojson(
        building_score,
        output_path=None,
        output_type='geojson',
        min_area=min_area_pix,
        bg_threshold=score_thresh,
        do_transform=None,
        simplify=simplify)

    if output_path is not None:
        if len(polygon_gdf) > 0:
            polygon_gdf.to_file(output_path, driver='GeoJSON')
        else:
            save_empty_geojson(output_path)

    return polygon_gdf


def gen_building_polys_using_watershed(building_score,
                                       seed_min_area_pix,
                                       min_area_pix,
                                       seed_score_thresh,
                                       main_score_thresh,
                                       output_path=None):
    """[summary]

    Args:
        building_score ([type]): [description]
        seed_min_area_pix ([type]): [description]
        min_area_pix ([type]): [description]
        seed_score_thresh ([type]): [description]
        main_score_thresh ([type]): [description]
        output_path ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    import geopandas as gpd
    import numpy as np
    from skimage import measure
    from skimage.morphology import watershed

    def remove_small_regions(pred, min_area):
        """[summary]

        Args:
            pred ([type]): [description]
            min_area ([type]): [description]

        Returns:
            [type]: [description]
        """
        from skimage import measure

        props = measure.regionprops(pred)
        for i in range(len(props)):
            if props[i].area < min_area:
                pred[pred == i + 1] = 0
        return measure.label(pred, connectivity=2, background=0)

    def mask_to_polys(mask):
        """[summary]

        Args:
            mask ([type]): [description]

        Returns:
            [type]: [description]
        """
        import pandas as pd
        from rasterio import features
        from shapely import ops, geometry

        shapes = features.shapes(y_pred.astype(np.int16), mask > 0)
        mp = ops.cascaded_union(
            geometry.MultiPolygon(
                [geometry.shape(shape) for shape, value in shapes]))

        if isinstance(mp, geometry.Polygon):
            polygon_gdf = pd.DataFrame({
                'geometry': [mp],
            })
        else:
            polygon_gdf = pd.DataFrame({
                'geometry': [p for p in mp],
            })
        return polygon_gdf

    av_pred = (building_score > seed_score_thresh).astype(np.uint8)
    y_pred = measure.label(av_pred, connectivity=2, background=0)
    y_pred = remove_small_regions(y_pred, seed_min_area_pix)

    nucl_msk = 1 - building_score
    nucl_msk = (nucl_msk * 65535).astype('uint16')
    y_pred = watershed(nucl_msk,
                       y_pred,
                       mask=(building_score > main_score_thresh),
                       watershed_line=True)
    y_pred = remove_small_regions(y_pred, min_area_pix)
    polygon_gdf = mask_to_polys(y_pred)

    if output_path is not None:
        if len(polygon_gdf) > 0:
            polygon_gdf = gpd.GeoDataFrame(polygon_gdf)
            polygon_gdf.to_file(output_path, driver='GeoJSON')
        else:
            save_empty_geojson(output_path)

    return polygon_gdf


def calculate_iou(pred_poly, test_data_GDF):
    """Get the best intersection over union for a predicted polygon.
    Adapted from: https://github.com/CosmiQ/solaris/blob/master/solaris/eval/iou.py, but
    keeps index of test_data_GDF
    Arguments
    ---------
    pred_poly : :py:class:`shapely.Polygon`
        Prediction polygon to test.
    test_data_GDF : :py:class:`geopandas.GeoDataFrame`
        GeoDataFrame of ground truth polygons to test ``pred_poly`` against.
    Returns
    -------
    iou_GDF : :py:class:`geopandas.GeoDataFrame`
        A subset of ``test_data_GDF`` that overlaps ``pred_poly`` with an added
        column ``iou_score`` which indicates the intersection over union value.
    """
    import geopandas as gpd

    # Fix bowties and self-intersections
    if not pred_poly.is_valid:
        pred_poly = pred_poly.buffer(0.0)

    precise_matches = test_data_GDF[test_data_GDF.intersects(pred_poly)]

    iou_row_list = []
    for idx, row in precise_matches.iterrows():
        # Load ground truth polygon and check exact iou
        test_poly = row.geometry
        # Ignore invalid polygons for now
        if pred_poly.is_valid and test_poly.is_valid:
            intersection = pred_poly.intersection(test_poly).area
            union = pred_poly.union(test_poly).area
            # Calculate iou
            iou_score = intersection / float(union)
            gt_idx = idx
        else:
            iou_score = 0
            gt_idx = -1
        row['iou_score'] = iou_score
        row['gt_idx'] = gt_idx
        iou_row_list.append(row)

    iou_GDF = gpd.GeoDataFrame(iou_row_list)
    return iou_GDF


def track_footprint_identifiers(json_dir,
                                out_dir,
                                min_iou=0.25,
                                iou_field='iou_score',
                                id_field='Id',
                                reverse_order=False,
                                verbose=True,
                                super_verbose=False):
    """Track footprint identifiers in the deep time stack.
    We need to track the global gdf instead of just the gdf of t-1.
    Args:
        json_dir ([type]): [description]
        out_dir ([type]): [description]
        min_iou (float, optional): [description]. Defaults to 0.25.
        iou_field (str, optional): [description]. Defaults to 'iou_score'.
        id_field (str, optional): [description]. Defaults to 'Id'.
        reverse_order (bool, optional): [description]. Defaults to False.
        verbose (bool, optional): [description]. Defaults to True.
        super_verbose (bool, optional): [description]. Defaults to False.

    Raises:
        Exception: [description]
        Exception: [description]
        Exception: [description]
        Exception: [description]
        Exception: [description]
    """
    import math
    import os

    import geopandas as gpd
    import numpy as np

    os.makedirs(out_dir, exist_ok=True)

    # set columns for master gdf
    gdf_master_columns = [id_field, iou_field, 'area', 'geometry']

    json_files = sorted([
        f for f in os.listdir(os.path.join(json_dir))
        if f.endswith('.geojson') and os.path.exists(os.path.join(json_dir, f))
    ])
    # start at the end and work backwards?
    if reverse_order:
        json_files = json_files[::-1]

    # check if only partical matching has been done (this will cause errors)
    out_files_tmp = sorted(
        [z for z in os.listdir(out_dir) if z.endswith('.geojson')])
    if len(out_files_tmp) > 0:
        if len(out_files_tmp) != len(json_files):
            raise Exception(
                "\nError in:", out_dir, "with N =", len(out_files_tmp),
                "files, need to purge this folder and restart matching!\n")
            return
        elif len(out_files_tmp) == len(json_files):
            print("\nDir:", os.path.basename(out_dir), "N files:",
                  len(json_files), "directory matching completed, skipping...")
            return
    else:
        print("\nMatching json_dir: ", os.path.basename(json_dir), "N json:",
              len(json_files))

    gdf_dict = {}
    for j, f in enumerate(json_files):

        name_root = f.split('.')[0]
        json_path = os.path.join(json_dir, f)
        output_path = os.path.join(out_dir, f)

        if verbose and ((j % 1) == 0):
            print("  ", j, "/", len(json_files), "for",
                  os.path.basename(json_dir), "=", name_root)

        # gdf
        gdf_now = gpd.read_file(json_path)
        # drop value if it exists
        gdf_now = gdf_now.drop(columns=['value'])
        # get area
        gdf_now['area'] = gdf_now['geometry'].area
        # initialize iou, id
        gdf_now[iou_field] = -1
        gdf_now[id_field] = -1
        # sort by reverse area
        gdf_now.sort_values(by=['area'], ascending=False, inplace=True)
        gdf_now = gdf_now.reset_index(drop=True)
        # reorder columns (if needed)
        gdf_now = gdf_now[gdf_master_columns]
        id_set = set([])

        if verbose:
            print("\n")
            print("", j, "file_name:", f)
            print("  ", "gdf_now.columns:", gdf_now.columns)

        if j == 0:
            # Establish initial footprints at Epoch0
            # set id
            gdf_now[id_field] = gdf_now.index.values
            gdf_now[iou_field] = 0
            n_new = len(gdf_now)
            n_matched = 0
            id_set = set(gdf_now[id_field].values)
            gdf_master_Out = gdf_now.copy(deep=True)
            # gdf_dict[f] = gdf_now
        else:
            # match buildings in epochT to epochT-1
            # see: https://github.com/CosmiQ/solaris/blob/master/solaris/eval/base.py
            # print("gdf_master;", gdf_dict['master']) #gdf_master)
            gdf_master_Out = gdf_dict['master'].copy(deep=True)
            gdf_master_Edit = gdf_dict['master'].copy(deep=True)

            if verbose:
                print("   len gdf_now:", len(gdf_now), "len(gdf_master):",
                      len(gdf_master_Out), "max master id:",
                      np.max(gdf_master_Out[id_field]))
                print("   gdf_master_Edit.columns:", gdf_master_Edit.columns)

            new_id = np.max(gdf_master_Edit[id_field]) + 1
            # if verbose:
            #    print("new_id:", new_id)
            idx = 0
            n_new = 0
            n_matched = 0
            for pred_idx, pred_row in gdf_now.iterrows():
                if verbose:
                    if (idx % 1000) == 0:
                        print("    ", name_root, idx, "/", len(gdf_now))
                if super_verbose:
                    # print("    ", i, j, idx, "/", len(gdf_now))
                    print("    ", idx, "/", len(gdf_now))
                idx += 1
                pred_poly = pred_row.geometry
                # if super_verbose:
                #     print("     pred_poly.exterior.coords:", list(pred_poly.exterior.coords))

                # get iou overlap
                iou_GDF = calculate_iou(pred_poly, gdf_master_Edit)
                # iou_GDF = iou.calculate_iou(pred_poly, gdf_master_Edit)
                # print("iou_GDF:", iou_GDF)

                # Get max iou
                if not iou_GDF.empty:
                    max_iou_row = iou_GDF.loc[iou_GDF['iou_score'].idxmax(
                        axis=0, skipna=True)]
                    # sometimes we are get an erroneous id of 0, caused by nan area,
                    #   so check for this
                    max_area = max_iou_row.geometry.area
                    if max_area == 0 or math.isnan(max_area):
                        # print("nan area!", max_iou_row, "returning...")
                        raise Exception("\n Nan area!:", max_iou_row,
                                        "returning...")
                        return

                    id_match = max_iou_row[id_field]
                    if id_match in id_set:
                        print("Already seen id! returning...")
                        raise Exception("\n Already seen id!", id_match,
                                        "returning...")
                        return

                    # print("iou_GDF:", iou_GDF)
                    if max_iou_row['iou_score'] >= min_iou:
                        if super_verbose:
                            print("    pred_idx:", pred_idx, "match_id:",
                                  max_iou_row[id_field], "max iou:",
                                  max_iou_row['iou_score'])
                        # we have a successful match, so set iou, and id
                        gdf_now.loc[pred_row.name,
                                    iou_field] = max_iou_row['iou_score']
                        gdf_now.loc[pred_row.name, id_field] = id_match
                        # drop  matched polygon in ground truth
                        gdf_master_Edit = gdf_master_Edit.drop(
                            max_iou_row.name, axis=0)
                        n_matched += 1
                        # # update gdf_master geometry?
                        # # Actually let's leave the geometry the same so it doesn't move around...
                        # gdf_master_Out.at[max_iou_row['gt_idx'], 'geometry'] = pred_poly
                        # gdf_master_Out.at[max_iou_row['gt_idx'], 'area'] = pred_poly.area
                        # gdf_master_Out.at[max_iou_row['gt_idx'], iou_field] = max_iou_row['iou_score']

                    else:
                        # no match,
                        if super_verbose:
                            print("    Minimal match! - pred_idx:", pred_idx,
                                  "match_id:", max_iou_row[id_field],
                                  "max iou:", max_iou_row['iou_score'])
                            print("      Using new id:", new_id)
                        if (new_id in id_set) or (new_id == 0):
                            raise Exception(
                                "trying to add an id that already exists, returning!"
                            )
                            return
                        gdf_now.loc[pred_row.name, iou_field] = 0
                        gdf_now.loc[pred_row.name, id_field] = new_id
                        id_set.add(new_id)
                        # update master, cols = [id_field, iou_field, 'area', 'geometry']
                        gdf_master_Out.loc[new_id] = [
                            new_id, 0, pred_poly.area, pred_poly
                        ]
                        new_id += 1
                        n_new += 1

                else:
                    # no match (same exact code as right above)
                    if super_verbose:
                        print("    pred_idx:", pred_idx, "no overlap, new_id:",
                              new_id)
                    if (new_id in id_set) or (new_id == 0):
                        raise Exception(
                            "trying to add an id that already exists, returning!"
                        )
                        return
                    gdf_now.loc[pred_row.name, iou_field] = 0
                    gdf_now.loc[pred_row.name, id_field] = new_id
                    id_set.add(new_id)
                    # update master, cols = [id_field, iou_field, 'area', 'geometry']
                    gdf_master_Out.loc[new_id] = [
                        new_id, 0, pred_poly.area, pred_poly
                    ]
                    new_id += 1
                    n_new += 1

        # print("gdf_now:", gdf_now)
        gdf_dict[f] = gdf_now
        gdf_dict['master'] = gdf_master_Out

        # save!
        if len(gdf_now) > 0:
            gdf_now.to_file(output_path, driver="GeoJSON")
        else:
            print("Empty dataframe, writing empty gdf", output_path)
            open(output_path, 'a').close()

        if verbose:
            print("  ", "N_new, N_matched:", n_new, n_matched)


def convert_geojsons_to_csv(json_dirs, output_csv_path, population='proposal'):
    """Convert jsons to csv
    Population is either "ground" or "proposal"
    Args:
        json_dirs ([type]): [description]
        output_csv_path ([type]): [description]
        population (str, optional): [description]. Defaults to 'proposal'.

    Raises:
        Exception: [description]
        Exception: [description]
        Exception: [description]

    Returns:
        [type]: [description]
    """
    import os.path
    from glob import glob

    import fiona
    import geopandas as gpd
    from tqdm import tqdm

    first_file = True  # switch that will be turned off once we process the first file
    for json_dir in tqdm(json_dirs):
        json_files = sorted(glob(os.path.join(json_dir, '*.geojson')))
        for json_file in tqdm(json_files):
            try:
                df = gpd.read_file(json_file)
            except (fiona.errors.DriverError):
                message = '! Invalid dataframe for %s' % json_file
                print(message)
                continue
                #raise Exception(message)
            if population == 'ground':
                file_name_col = df.image_fname.apply(
                    lambda x: os.path.splitext(x)[0])
            elif population == 'proposal':
                file_name_col = os.path.splitext(
                    os.path.basename(json_file))[0]
            else:
                raise Exception('! Invalid population')
            df = gpd.GeoDataFrame({
                'filename': file_name_col,
                'id': df.Id.astype(int),
                'geometry': df.geometry,
            })
            if len(df) == 0:
                message = '! Empty dataframe for %s' % json_file
                print(message)
                #raise Exception(message)

            if first_file:
                net_df = df
                first_file = False
            else:
                net_df = net_df.append(df)

    net_df.to_csv(output_csv_path, index=False)
    return net_df
