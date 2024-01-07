from enum import Enum
from typing import List
import numpy as np
import pandas as pd
from typing import Tuple

from src.eval.filters.lanes.src.adapter import Lanes, Lane


class BoundariesLabel(Enum):
    NOT_DECIDED = -1
    OUTSIDE_BOUNDARIES = 0
    INSIDE_BOUNDARIES = 1
    INSIDE_BOUNDARIES_MARGIN = 2


def _is_inside_decision_region(detections: pd.DataFrame,
                               lanes: Lanes,
                               excess_limit_top: float,
                               excess_limit_bottom: float):
    bottom_y = (detections.y_center + detections.height / 2).values

    is_in_left_region = np.ones_like(len(detections), dtype=bool)
    is_in_right_region = np.ones_like(len(detections), dtype=bool)

    if lanes.road_boundary_left is not None:
        is_in_left_region = np.logical_and(bottom_y > lanes.road_boundary_left.active_region[0] - excess_limit_top,
                                           bottom_y < lanes.road_boundary_left.active_region[1] + excess_limit_bottom)

    if lanes.road_boundary_right is not None:
        is_in_right_region = np.logical_and(bottom_y > lanes.road_boundary_right.active_region[0] - excess_limit_top,
                                            bottom_y < lanes.road_boundary_right.active_region[1] + excess_limit_bottom)

    return is_in_left_region, is_in_right_region


def _is_outside_boundary(detections: pd.DataFrame,
                         boundary_lane: Lane,
                         distance_ths: float,
                         inside_ratio_ths: float,
                         excess_limit_top: float,
                         excess_limit_bottom: float,
                         adaptive_margin: Tuple[float, float] = (0, 0),
                         side='left'):

    is_outside_boundary = np.zeros_like(len(detections), dtype=bool)

    if boundary_lane is None:
        return is_outside_boundary

    is_inside_decision_region = np.zeros_like(len(detections), dtype=bool)
    distance_from_boundary = np.empty_like(len(detections), dtype=bool)
    inside_boundary_ratio = np.empty_like(len(detections), dtype=float)

    s = 1 if side == 'left' else -1

    adaptive_margin_fn = np.poly1d(np.polyfit(np.array([boundary_lane.active_region[0], boundary_lane.active_region[1]]),
                                              np.array([adaptive_margin[0], adaptive_margin[1]]), 1))

    bottom_y = (detections.y_center + detections.height / 2).values

    bottom_corner_x = (
        detections.x_center + s*detections.width / 2).values

    margin_y = adaptive_margin_fn(bottom_y)
    margin_distance = margin_y + distance_ths

    distance_from_boundary = s * \
        (boundary_lane(bottom_y) - s*margin_distance - bottom_corner_x)
    inside_boundary_ratio = np.clip(
        -distance_from_boundary / detections.width, 0, 1)

    is_outside_upper_boundary = np.logical_and(
        bottom_y < boundary_lane.active_region[0] - excess_limit_top,
        s*(boundary_lane(boundary_lane.active_region[0]) - bottom_corner_x) > 0)

    is_inside_decision_region = np.logical_and(bottom_y > boundary_lane.active_region[0] - excess_limit_top,
                                               bottom_y < boundary_lane.active_region[1] + excess_limit_bottom)

    is_outside_boundary = np.logical_or(is_outside_upper_boundary, np.logical_and(
        inside_boundary_ratio < inside_ratio_ths, is_inside_decision_region))

    return is_outside_boundary


def are_outside_road_boundaries(detections: pd.DataFrame,
                                lanes: Lanes,
                                distance_ths: float = 0,
                                margin: Tuple[float, float] = (0, 0),
                                inside_ratio_ths: float = 1,
                                excess_limit_top: float = 0,
                                excess_limit_bottom: float = 0) -> List[BoundariesLabel]:
    """
    Determines whether a detection is outside road boundaries.
    This logic rules out detections based on the left and right road boundaries.
    If the bottom corner of a detection is on the 'wrong' side of the road boundary, it is considered as out of boundary.
    A detection can be considered inside road boundaries only if it is also inside the decision region for both left and right road boundaries.
    Any detection which doesnt fall under the inside or outside rules, considered to be 'NOT DECIDIED'.

    Args:
        detections (pd.DataFrame): Detections
        lanes (Lanes): Road Lanes

    Returns:
        List[BoundariesLabel]: list of boundary labels. Each detection produces a boundary decision.
    """

    labels = np.full(
        len(detections), fill_value=BoundariesLabel.NOT_DECIDED)

    is_outside_left_boundary = _is_outside_boundary(detections=detections,
                                                    boundary_lane=lanes.road_boundary_left,
                                                    distance_ths=distance_ths,
                                                    inside_ratio_ths=inside_ratio_ths,
                                                    excess_limit_top=excess_limit_top,
                                                    excess_limit_bottom=excess_limit_bottom,
                                                    side='left')
    is_outside_left_boundary_with_margin = _is_outside_boundary(detections=detections,
                                                                boundary_lane=lanes.road_boundary_left,
                                                                distance_ths=distance_ths,
                                                                inside_ratio_ths=inside_ratio_ths,
                                                                excess_limit_top=excess_limit_top,
                                                                excess_limit_bottom=excess_limit_bottom,
                                                                adaptive_margin=margin,
                                                                side='left')

    is_outside_right_boundary = _is_outside_boundary(detections=detections,
                                                     boundary_lane=lanes.road_boundary_right,
                                                     distance_ths=distance_ths,
                                                     inside_ratio_ths=inside_ratio_ths,
                                                     excess_limit_top=excess_limit_top,
                                                     excess_limit_bottom=excess_limit_bottom,
                                                     side='right')
    is_outside_right_boundary_with_margin = _is_outside_boundary(detections=detections,
                                                                 boundary_lane=lanes.road_boundary_right,
                                                                 distance_ths=distance_ths,
                                                                 inside_ratio_ths=inside_ratio_ths,
                                                                 excess_limit_top=excess_limit_top,
                                                                 excess_limit_bottom=excess_limit_bottom,
                                                                 adaptive_margin=margin,
                                                                 side='right')

    is_outside_boundaries = np.logical_or(
        is_outside_left_boundary, is_outside_right_boundary)
    labels[is_outside_boundaries] = BoundariesLabel.OUTSIDE_BOUNDARIES

    is_in_left_region, is_in_right_region = _is_inside_decision_region(detections=detections,
                                                                       lanes=lanes,
                                                                       excess_limit_bottom=excess_limit_bottom,
                                                                       excess_limit_top=excess_limit_top)
    labels[~is_outside_left_boundary_with_margin * is_in_left_region *
           ~is_outside_right_boundary_with_margin * is_in_right_region] = BoundariesLabel.INSIDE_BOUNDARIES_MARGIN
    labels[~is_outside_left_boundary * is_in_left_region *
           ~is_outside_right_boundary * is_in_right_region] = BoundariesLabel.INSIDE_BOUNDARIES

    return labels, None


def are_detections_above_horizon(detections: pd.DataFrame,
                                 horizon: np.polynomial.Polynomial,
                                 distance_from_horizon_threshold: float) -> List[bool]:
    """
    Based on given gorizon and defined distance from conf file, determine whether a detection is above the horizon.

    Args:
        detections (pd.DataFrame): Detections DataFrame
        horizon (Polynomial): a line which represents the horizon.

    Returns:
        _type_: Binary array whether a detection is above the horizon.
    """

    bottom_y = (detections.y_center + detections.height / 2).values
    distance_from_horizon = np.array(horizon(detections.x_center) - bottom_y)
    are_under_horizon = distance_from_horizon > distance_from_horizon_threshold

    return np.array(are_under_horizon), distance_from_horizon
