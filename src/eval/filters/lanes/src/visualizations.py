import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from typing import List, Tuple
from PIL import Image
import logging

from src.eval.filters.lanes.src.adapter import Lanes
from src.eval.filters.lanes.src.corridor import LanesCorridor
from src.eval.filters.lanes.src.road_boundaries import are_outside_road_boundaries, BoundariesLabel
from src.eval.filters.lanes.src.lane_assignment import RelativeLanePosition

logger = logging.getLogger('auto_infer')


BOUNDARY_TO_COLOR = {
    "ego_right": "darkgreen",
    "ego_left": "darkgreen",
    "adjacent_right": "blue",
    "adjacent_left": "blue",
    "road_boundary_right": "red",
    "road_boundary_left": "red",
}

LANE_ASSIGNMENT_TO_COLOR = {
    RelativeLanePosition.CIPV.name: 'purple',
    RelativeLanePosition.EGO.name: 'darkgreen',
    RelativeLanePosition.NEXT_LANE_LEFT.name: 'darkgreen',
    RelativeLanePosition.NEXT_LANE_RIGHT.name: 'darkgreen',
    RelativeLanePosition.LEFT.name: 'blue',
    RelativeLanePosition.RIGHT.name: 'blue',
    RelativeLanePosition.NOT_DECIDED.name: 'grey'
}


def get_valid_mask(xs, ys, image_size):
    valid = (ys > 0) * (ys < image_size[1])
    valid *= (xs > 0) * (xs < image_size[0])
    return valid


def draw_lanes(ax: plt.Axes, lanes: Lanes, image_size: Tuple, with_polynoms=False):
    for lane in lanes:
        valid = get_valid_mask(
            lane.points_xs, lane.points_ys, image_size=image_size)
        ax.scatter(lane.points_xs[valid], lane.points_ys[valid],
                   color=BOUNDARY_TO_COLOR[lane.boundary_name],
                   linewidth=1,
                   alpha=0.7,
                   s=3,
                   label=lane.boundary_name)

        if with_polynoms:
            ys = np.arange(0, image_size[1], 50)
            xs = lane(ys)
            valid = get_valid_mask(xs=xs, ys=ys, image_size=image_size)
            ax.plot(xs[valid], ys[valid],
                    linewidth=1, alpha=0.5)


def draw_detections(ax: plt.Axes,
                    detections: pd.DataFrame,
                    is_outside_boundaries: pd.Series,
                    lane_types: pd.Series):
    """
    Draws detections on existing axis. 
    Assign colors and marks based on boundaries labels and lane types.
    """

    appeared_labels = []
    for i, row in detections.reset_index().iterrows():
        x = row.x_center - row.width / 2
        y = row.y_center - row.height / 2

        if is_outside_boundaries.iloc[i] == BoundariesLabel.OUTSIDE_BOUNDARIES.name:
            color = 'red'
            label = BoundariesLabel.OUTSIDE_BOUNDARIES.name

        elif is_outside_boundaries.iloc[i] == BoundariesLabel.INSIDE_BOUNDARIES_MARGIN.name:
            if detections.iloc[i].label == 0:
                color = 'darkorange'
                label = BoundariesLabel.INSIDE_BOUNDARIES_MARGIN.name
            else:
                color = 'red'
                label = BoundariesLabel.OUTSIDE_BOUNDARIES.name

        else:
            color = LANE_ASSIGNMENT_TO_COLOR[lane_types.iloc[i]]
            label = lane_types.iloc[i]

        if label not in appeared_labels:
            appeared_labels.append(label)
        else:
            label = None

        ax.add_patch(Rectangle((x, y), row.width, row.height, alpha=1,
                     fill=None, edgecolor=color, label=label))


def draw_boundaries_area(ax: plt.Axes, cls: LanesCorridor, lanes: Lanes, image_size: Tuple):
    """
    Draws in boundries area under given curves.
    """

    if len(lanes) == 0:
        return

    x_center, y_center = np.meshgrid(np.arange(
        0, image_size[0], 50), np.arange(0, image_size[1], 50), indexing='ij')
    x_center, y_center = np.concatenate(x_center), np.concatenate(y_center)
    mock_detections = pd.DataFrame(np.array([x_center, y_center, np.zeros_like(x_center), np.zeros_like(x_center)]).T,
                                   columns=['x_center', 'y_center', 'height', 'width'])
    are_out_of_boundaries, _ = are_outside_road_boundaries(detections=mock_detections,
                                                           lanes=lanes,
                                                           excess_limit_top=cls.params.road_boundary_excess_limit_top,
                                                           excess_limit_bottom=cls.params.road_boundary_excess_limit_bottom,
                                                           distance_ths=cls.params.distance_from_road_boundary_threshold,
                                                           inside_ratio_ths=cls.params.inside_road_boundary_ratio_threshold,
                                                           margin=cls.params.distance_from_road_boundary_margin)
    mock_detections['rbs_label'] = [x.name for x in are_out_of_boundaries]

    vis_args = {'s': 2, 'marker': '2', 'alpha': 0.5}

    inside = mock_detections[mock_detections.rbs_label ==
                             BoundariesLabel.INSIDE_BOUNDARIES.name]
    ax.scatter(inside.x_center, inside.y_center, c='green', **vis_args)

    outside = mock_detections[mock_detections.rbs_label ==
                              BoundariesLabel.INSIDE_BOUNDARIES_MARGIN.name]
    ax.scatter(outside.x_center, outside.y_center, c='orange', **vis_args)

    outside = mock_detections[mock_detections.rbs_label ==
                              BoundariesLabel.OUTSIDE_BOUNDARIES.name]
    ax.scatter(outside.x_center, outside.y_center, c='red', **vis_args)

    not_decided = mock_detections[mock_detections.rbs_label ==
                                  BoundariesLabel.NOT_DECIDED.name]
    ax.scatter(not_decided.x_center,
               not_decided.y_center, c='grey', **vis_args)


def draw_horizon(ax: plt.Axes, horizon: np.polynomial.Polynomial, image_size: Tuple):
    xs = np.array([0, image_size[0]])
    ax.plot(xs, horizon(xs), label='horizon',
            color='darkgrey', linestyle='--', alpha=0.3)


def visualize_lanes_corridor(frame_id: int,
                             detections: pd.DataFrame,
                             images_dirpath: str,
                             lanes: Lanes,
                             horizon: np.polynomial.Polynomial,
                             lanes_corridor: LanesCorridor,
                             plot=True,
                             save_path=False
                             ):
    print(f'Plotting frame {frame_id}')

    image_name = f'{frame_id}.png'
    image_path = Path(images_dirpath).joinpath(image_name)
    try:
        img = Image.open(image_path)
    except Exception as e:
        logger.warning(f'No image found in {image_path}')
        return

    _, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.set_title(image_name)
    ax.imshow(img)

    draw_detections(ax=ax,
                    detections=detections,
                    is_outside_boundaries=detections.is_outside_boundaries,
                    lane_types=detections.lane_assignment)
    draw_lanes(ax=ax,
               lanes=lanes,
               image_size=img.size)
    draw_horizon(ax=ax,
                 horizon=horizon,
                 image_size=img.size)
    draw_boundaries_area(ax=ax,
                         cls=lanes_corridor,
                         lanes=lanes,
                         image_size=img.size)

    plt.legend(fontsize='large', loc='upper right')
    plt.tight_layout()

    if save_path:
        save_plot_path = Path(save_path)
        if not os.path.exists(save_plot_path):
            os.makedirs(Path(save_plot_path))

        img_save_path = Path(save_plot_path).joinpath(
            f'{image_name}_out.png')
        plt.savefig(img_save_path)

    if plot:
        plt.show()

    plt.close()
