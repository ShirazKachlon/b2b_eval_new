import pandas as pd
import numpy as np
from enum import Enum
from typing import List

from src.eval.filters.lanes.src.adapter import Lanes


class RelativeLanePosition(Enum):
    NOT_DECIDED = -1
    EGO = 0
    NEXT_LANE_LEFT = 1
    NEXT_LANE_RIGHT = 2
    LEFT = 3
    RIGHT = 4
    CIPV = 10
    MCP = 20


def _find_detections_cipv(detections: pd.DataFrame, lane_types: List[RelativeLanePosition]) -> int:
    if RelativeLanePosition.EGO not in lane_types:
        return -1

    is_veichle = (detections.label == '4w').values  # TODO: change to wider range of labels based on config
    is_in_ego = lane_types == RelativeLanePosition.EGO
    is_veichle_in_ego = np.logical_and(is_in_ego, is_veichle)

    if not np.any(is_veichle_in_ego):
        return -1

    ego_indices = np.argwhere(is_veichle_in_ego)

    bottom_y = (detections.y_center + detections.height / 2).values
    cipv_idx = ego_indices[np.argmax(bottom_y[ego_indices])][0]

    return cipv_idx


def _find_general_lane_assignment(detections: pd.DataFrame, lanes: Lanes) -> List[RelativeLanePosition]:
    # If there are less than 2 lanes, can't assign lanes
    if len(lanes) < 2:
        return np.full(len(detections), fill_value=RelativeLanePosition.NOT_DECIDED)

    lane_types = np.full(
        len(detections), fill_value=RelativeLanePosition.EGO)

    bottom_y = (detections.y_center + detections.height / 2).values

    if lanes.ego_left is not None:
        are_left_to_ego_left = lanes.ego_left(
            bottom_y) - detections.x_center > 0
        lane_types[are_left_to_ego_left] = RelativeLanePosition.NEXT_LANE_LEFT

    if lanes.adjacent_left is not None:
        are_left_to_adjacent_left = lanes.adjacent_left(
            bottom_y) - detections.x_center > 0
        lane_types[are_left_to_adjacent_left] = RelativeLanePosition.LEFT

    if lanes.road_boundary_left is not None:
        are_left_to_road_boundary_left = lanes.road_boundary_left(
            bottom_y) - detections.x_center > 0
        lane_types[are_left_to_road_boundary_left] = RelativeLanePosition.LEFT

    if lanes.ego_right is not None:
        are_right_to_ego_right = detections.x_center - \
            lanes.ego_right(bottom_y) > 0
        lane_types[are_right_to_ego_right] = RelativeLanePosition.NEXT_LANE_RIGHT

    if lanes.adjacent_right is not None:
        are_right_to_adjacent_right = detections.x_center - \
            lanes.adjacent_right(bottom_y) > 0
        lane_types[are_right_to_adjacent_right] = RelativeLanePosition.RIGHT

    if lanes.road_boundary_right is not None:
        are_right_to_road_boundary_right = detections.x_center - \
            lanes.road_boundary_right(bottom_y) > 0
        lane_types[are_right_to_road_boundary_right] = RelativeLanePosition.RIGHT

    if lanes.ego_left is not None and lanes.ego_right is not None:
        are_in_current_lane = np.logical_and(
            ~are_left_to_ego_left, ~are_right_to_ego_right)
        lane_types[are_in_current_lane] = RelativeLanePosition.EGO

    return lane_types


def find_detections_angle(detections: pd.DataFrame) -> List[float]:
    return np.zeros(len(detections))


def find_detections_lane_type(detections: pd.DataFrame, lanes: Lanes) -> List[RelativeLanePosition]:
    lane_types = _find_general_lane_assignment(detections=detections,
                                               lanes=lanes)

    cipv_idx = _find_detections_cipv(detections=detections,
                                     lane_types=lane_types)
    if cipv_idx > -1:
        lane_types[cipv_idx] = RelativeLanePosition.CIPV

    return lane_types
