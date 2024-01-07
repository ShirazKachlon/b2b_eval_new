from time import time
import multiprocessing as mp
from tqdm import tqdm
from pathlib import Path
import os
import logging

import pandas as pd

from filters.lanes.src.corridor import LanesCorridor, LanesCorridorParams

logger: logging.Logger = logging.getLogger("lanes")


def apply_lanes_attributes(cametra_interface_data_path: str,
                           b2b_pred_path: str,
                           save_path: str = None,
                           plot: bool = False,
                           images_dir_path: str = None,
                           save_plots_path: str = None,
                           multiprocess=False):

    pred_df = pd.read_csv(b2b_pred_path, sep='\t')

    start_time = time()

    params = LanesCorridorParams(cametra_data_path=cametra_interface_data_path,
                                 images_path=images_dir_path,
                                 plot=plot,
                                 save_plots=save_plots_path)
    rbs = LanesCorridor(params=params)

    rbs.apply_all(detections=pred_df,
                  save_path=save_path,
                  multiprocess=multiprocess)

    logger.info(f'Task took {time() - start_time} seconds in total.')


def _process_set(set_path: Path, result_queue: mp.Queue = None):
    from filters.lanes.src.corridor import LanesCorridor, LanesCorridorParams

    b2b_pred_path = set_path.joinpath('results/infer/b2b_det.tsv')
    prod_pred_path = set_path.joinpath('prod_det_sf.tsv')
    cametra_interface_data_path = set_path

    try:
        b2b_det_df = pd.read_csv(b2b_pred_path, sep='\t')
        prod_pred_df = pd.read_csv(prod_pred_path, sep='\t')

        params = LanesCorridorParams(
            cametra_data_path=cametra_interface_data_path)
        rbs = LanesCorridor(params=params)

        results_b2b_det_df = rbs.apply_all(
            detections=b2b_det_df, save_path=b2b_pred_path.parent)

        prod_pred_df.name = [
            str(x) + '.png' for x in prod_pred_df.name]
        results_prod_det_df = rbs.apply_all(
            detections=prod_pred_df, save_path=prod_pred_path.parent)

        if result_queue:
            result_queue.put(
                (set_path, results_b2b_det_df, results_prod_det_df))

    except Exception as e:
        logger.warning(f'Failed on {set_path}: {e}')
        results_b2b_det_df = None
        if result_queue:
            result_queue.put((set_path, None, None))

    logger.info(f'Finished processing {set_path}')


def apply_lanes_attributes_on_set(set_path: str):
    try:
        _process_set(Path(set_path), result_queue=None)
    except Exception as e:
        logger.error(e)


def apply_lanes_attributes_on_superset(set_path: str, save_path: str):

    subsets_ids = os.listdir(set_path)
    subsets_paths = [Path(set_path).joinpath(ssp) for ssp in subsets_ids]

    logger.info(f'Running on total of {len(subsets_ids)} sets: {subsets_ids}')

    num_processes = min(mp.cpu_count(), len(subsets_ids))
    logger.info(f'Run on total of {num_processes} cpus.')

    with mp.Pool(processes=num_processes) as pool:
        # Create a Queue to store the results

        manager = mp.Manager()
        result_queue = manager.Queue()

        pool.starmap(_process_set, [(subset, result_queue)
                                    for subset in subsets_paths])

        # Close the pool to indicate that no more tasks will be added
        pool.close()

        # Wait for all processes to finish
        pool.join()

        # Retrieve results from the queue
        results = []
        for _ in tqdm(subsets_paths):
            results.append(result_queue.get())

    failed_sets = [result[0] for result in results if result[1] is None]
    if len(failed_sets) > 0:
        logger.warn(
            f'Failed on {len(failed_sets)} sets out of {len(subsets_ids)} sets. Failed sets are {failed_sets}')
    logger.info(f'Success for all {len(subsets_ids)-len(failed_sets)} sets.')

    if len(results) > 0:
        b2b_results_df = pd.concat(
            [subset_df[1] for subset_df in results if subset_df[1] is not None])
        prod_results_df = pd.concat(
            [subset_df[2] for subset_df in results if subset_df[2] is not None])
    else:
        raise Exception('No results.')

    if not os.path.exists(Path(save_path)):
        os.makedirs(Path(save_path))

    csv_save_path = Path(save_path).joinpath(
        'superset_b2b_results_with_rbs.tsv')
    b2b_results_df.to_csv(csv_save_path, sep='\t', index=False)
    logger.info(
        f'Saved b2b results of shape {b2b_results_df.shape} to {csv_save_path}')

    csv_save_path = Path(save_path).joinpath(
        'superset_prod_results_with_rbs.tsv')
    prod_results_df.to_csv(csv_save_path, sep='\t', index=False)
    logger.info(
        f'Saved prod results of shape {prod_results_df.shape} to {csv_save_path}')
