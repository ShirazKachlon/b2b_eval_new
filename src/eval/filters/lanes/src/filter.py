import pandas as pd
import numpy as np
import json
from typing import List, Tuple
import logging
import traceback
from pathlib import Path

from src.eval.filters.lanes.src.corridor import LanesCorridor, LanesCorridorParams
from src.eval.filters.lanes.src.road_boundaries import BoundariesLabel
from src.eval.filters.lanes.src.lane_assignment import RelativeLanePosition


class UnknownClassException(Exception):
    pass


logger = logging.getLogger('evaluation')


def _load_json_config(eval_config: dict, config_name: str):
    config_path = next(
        c for c in eval_config['configs'] if c['config_name'] == config_name)['evaluation_configuration']
    with open(config_path) as f:
        config = json.load(f)
    return config


def _get_det_and_gt_classes_from_config(eval_config: dict, config_name: str):
    config = _load_json_config(eval_config, config_name)
    det_classes = config['data_config']['det_classes']
    gt_classes = config['data_config']['gt_classes']
    gt_classes_ignore = config['data_config']['gt_classes_ignore']
    return det_classes, gt_classes, gt_classes_ignore


def _get_veichles_to_ignore(df_wl: pd.DataFrame, classes: List[str], is_gt=True):
    """Basic veichle filter which filters any veichle which is not inside boundary or is not in close lanes which are EGO and NEXT LANES (right and left).
    """
    is_in_classes = df_wl.label.isin(classes)
    not_inside = df_wl.is_outside_boundaries != BoundariesLabel.INSIDE_BOUNDARIES.name

    to_ignore = not_inside
    if is_gt:
        in_close_lanes = df_wl.lane_assignment.isin([RelativeLanePosition.CIPV.name,
                                                    RelativeLanePosition.EGO.name,
                                                    RelativeLanePosition.NEXT_LANE_LEFT.name,
                                                    RelativeLanePosition.NEXT_LANE_RIGHT.name])

        to_ignore = np.logical_or(not_inside, ~in_close_lanes)

    return np.logical_and(is_in_classes, to_ignore)


def _get_peds_to_ignore(df_wl: pd.DataFrame, classes: List[str]):
    is_in_classes = df_wl.label.isin(classes)
    is_inside_boundaries = np.logical_or(df_wl.is_outside_boundaries == BoundariesLabel.INSIDE_BOUNDARIES.name,
                                         df_wl.is_outside_boundaries == BoundariesLabel.INSIDE_BOUNDARIES_MARGIN.name)

    return np.logical_and(is_in_classes, ~is_inside_boundaries)


def _get_unknown_to_ignore(df_wl: pd.DataFrame, classes: List[str]):
    is_in_classes = df_wl.label.isin(classes)
    not_inside = df_wl.is_outside_boundaries != BoundariesLabel.INSIDE_BOUNDARIES.name
    return np.logical_and(is_in_classes, not_inside)


def _apply_on_dataframe(df: pd.DataFrame,
                        lanes_corridor: LanesCorridor,
                        peds_classes: List[str],
                        rider_classes: List[str],
                        two_wheels_classes: List[str],
                        four_wheels_classes: List[str],
                        ignore_classes: List[str] = [],
                        is_gt: bool = True) -> Tuple[pd.DataFrame, dict]:

    df_classes = df.label.unique()
    known_classes = peds_classes + rider_classes + \
        two_wheels_classes + four_wheels_classes + ignore_classes
    unknown_classes = set(df_classes) - set(known_classes)
    # if is_gt and len(unknown_classes) > 0:
    #     raise UnknownClassException(
    #         f'Dataframe contains unknown classes: {unknown_classes}')

    try:
        df_wl, lanes_dict = lanes_corridor.apply(df)
        df_wl = df_wl.sort_index()

        veichles_to_ignore = _get_veichles_to_ignore(df_wl=df_wl,
                                                     classes=rider_classes + two_wheels_classes + four_wheels_classes,
                                                     is_gt=is_gt)
        peds_to_ignore = _get_peds_to_ignore(df_wl=df_wl,
                                             classes=peds_classes)

        # Fow now handle unknwon classes, should be removed.
        unkwnon_to_ignore = _get_unknown_to_ignore(df_wl=df_wl,
                                                   classes=unknown_classes)

        # General ignore logic, for example, when not both road boundaries are found.
        general_ignore = df_wl.found_both_boundaries.values == False

        # Sum up all ignores
        to_ignore = np.any([veichles_to_ignore, peds_to_ignore, unkwnon_to_ignore, general_ignore], axis=0)
    except Exception as e:
        logger.error(
            f'Error applying lanes filter: {e}, \n traceback: {traceback.format_exc()}')
        to_ignore = np.zeros(len(df), dtype=bool)

    is_already_ignored = df.label.isin(ignore_classes).values

    to_change = np.logical_and(to_ignore, ~is_already_ignored)
    lbls = [l if not to_change[i]
            else f'{l}_ignore' for i, l in enumerate(df_wl.label.values)]
    df_wl['label'] = lbls
    df_wl['lanes_ignore'] = to_ignore

    return df_wl, lanes_dict


def apply_lanes_filter(gt_df: pd.DataFrame,
                       det_df: pd.DataFrame,
                       cametra_data_path: Path,
                       eval_config: dict) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:

    params = LanesCorridorParams(cametra_data_path=cametra_data_path)
    lanes_corridor = LanesCorridor(params=params)

    peds_det_classes, peds_gt_classes, peds_gt_classes_ignore = _get_det_and_gt_classes_from_config(
        eval_config, 'peds')
    rider_det_classes, rider_gt_classes, rider_gt_classes_ignore = [], [], []  # TODO: fix
    # rider_det_classes, rider_gt_classes, rider_gt_classes_ignore = _get_det_and_gt_classes_from_config(
    #     eval_config, 'rider')
    two_wheels_det_classes, two_wheels_gt_classes, two_wheels_gt_classes_ignore = _get_det_and_gt_classes_from_config(
        eval_config, '2w')
    four_wheels_det_classes, four_wheels_gt_classes, four_wheels_gt_classes_ignore = _get_det_and_gt_classes_from_config(
        eval_config, '4w')
    gt_ignore_classes = peds_gt_classes_ignore + rider_gt_classes_ignore + \
        two_wheels_gt_classes_ignore + four_wheels_gt_classes_ignore

    old_dist_ths = lanes_corridor.params.distance_from_road_boundary_threshold
    new_dist_ths = lanes_corridor.params.distance_from_road_boundary_threshold + 20
    lanes_corridor.set_distance_from_road_boundary_threshold(new_dist_ths)

    updated_det_df, lanes_dict = _apply_on_dataframe(df=det_df,
                                                     lanes_corridor=lanes_corridor,
                                                     peds_classes=peds_det_classes,
                                                     rider_classes=rider_det_classes,
                                                     two_wheels_classes=two_wheels_det_classes,
                                                     four_wheels_classes=four_wheels_det_classes,
                                                     is_gt=False)

    lanes_corridor.set_distance_from_road_boundary_threshold(old_dist_ths)
    updated_gt_df, lanes_dict = _apply_on_dataframe(df=gt_df,
                                                    lanes_corridor=lanes_corridor,
                                                    peds_classes=peds_gt_classes,
                                                    rider_classes=rider_gt_classes,
                                                    two_wheels_classes=two_wheels_gt_classes,
                                                    four_wheels_classes=four_wheels_gt_classes,
                                                    ignore_classes=gt_ignore_classes)

    return updated_det_df, updated_gt_df, lanes_dict
