from pathlib import Path
import os
import logging
import traceback
from typing import Tuple

import pandas as pd
import numpy as np
import yaml
import json

from src.eval.filters.lanes.src.adapter import LanesModelAdapter, NoCametraDataException
from src.eval.filters.lanes.src.lane_assignment import find_detections_lane_type, RelativeLanePosition
from src.eval.filters.lanes.src.road_boundaries import are_outside_road_boundaries, are_detections_above_horizon, BoundariesLabel

logger: logging.Logger = logging.getLogger("lanes")


class LanesCorridorParams:
    def __init__(self,
                 cametra_data_path: str,
                 images_path=None,
                 conf_path=None,
                 plot=False,
                 save_plots: str = None) -> None:
        self.cametra_data_path = cametra_data_path
        self.image_dataset_path = images_path
        self.plot = plot
        self.save_plots = save_plots

        if conf_path is None:
            conf_path = Path(__name__).parent.joinpath('conf.yaml')

        with open(Path(__file__).parent.joinpath('conf.yaml')) as conf_f:
            conf = yaml.safe_load(conf_f)

        self.distance_from_road_boundary_threshold = conf['distance_from_road_boundary_threshold']
        self.distance_from_horizon_threshold = conf['distance_from_horizon_threshold']
        self.road_boundary_excess_limit_top = conf['road_boundary_excess_limit_top']
        self.road_boundary_excess_limit_bottom = conf['road_boundary_excess_limit_bottom']
        self.inside_road_boundary_ratio_threshold = conf['inside_road_boundary_ratio_threshold']
        self.distance_from_road_boundary_margin = tuple(conf['distance_from_road_boundary_margin'])


class LanesCorridor:
    def __init__(self, params: LanesCorridorParams) -> None:
        self.params = params

        self.adapter = LanesModelAdapter(
            cametra_data_path=self.params.cametra_data_path)

    def _get_lanes(self, frame_id: int):
        lanes = self.adapter.get_lanes(frame_id=frame_id)

        return lanes

    def _apply_single(self, frame_detections: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """
        This is the main module function.
        For given detections and frame id, apply out of boundary, above horizon and lane assignment logic.

        Args:
            detections (pd.DataFrame): Detections.

        Returns:
            Tuple[List[BoundariesLabel], List[bool], List[RelativeLanePosition]]: returns out of boundary decisions, above horizon and relative lanes posisions.
        """

        if len(frame_detections) == 0:
            return None

        assert np.all([c in frame_detections.columns for c in ['x_center', 'y_center', 'height', 'width']]
                      ), "Invalid DataFrame. Should contain columns: 'x_center', 'y_center', 'height', 'width'."

        assert self.adapter is not None, "Please set multiframe adapter using set_MF_adapter before applying this method."

        assert len(frame_detections.name.unique()
                   ) > 0, "This function accept detection for a single frame."

        image_name = frame_detections.name.unique()[0]
        frame_id = int(
            image_name[:-4]) if isinstance(image_name, str) else image_name

        # Get lanes and horizon from adapter
        lanes = self.adapter.get_lanes(frame_id=frame_id)
        horizon = self.adapter.get_horizon(frame_id=frame_id)

        are_above_horizon, _ = are_detections_above_horizon(detections=frame_detections,
                                                            horizon=horizon,
                                                            distance_from_horizon_threshold=self.params.distance_from_horizon_threshold)

        is_outside_boundaries, _ = are_outside_road_boundaries(detections=frame_detections,
                                                               lanes=lanes,
                                                               distance_ths=self.params.distance_from_road_boundary_threshold,
                                                               margin=self.params.distance_from_road_boundary_margin,
                                                               excess_limit_top=self.params.road_boundary_excess_limit_top,
                                                               excess_limit_bottom=self.params.road_boundary_excess_limit_bottom,
                                                               inside_ratio_ths=self.params.inside_road_boundary_ratio_threshold)

        detections_lane_type = find_detections_lane_type(
            detections=frame_detections, lanes=lanes)

        det_df = frame_detections.copy()
        det_df['is_outside_boundaries'] = [
            x.name for x in is_outside_boundaries]
        det_df['lane_assignment'] = [
            x.name for x in detections_lane_type]
        det_df['is_above_horizon'] = are_above_horizon
        det_df['found_both_boundaries'] = lanes.has_both_boundaries()

        lanes_dict = lanes.to_dict()

        if self.params.plot or self.params.save_plots:
            from src.eval.filters.lanes.src.visualizations import visualize_lanes_corridor

            visualize_lanes_corridor(frame_id=frame_id,
                                     detections=det_df,
                                     lanes=lanes,
                                     horizon=horizon,
                                     plot=self.params.plot,
                                     save_path=self.params.save_plots,
                                     lanes_corridor=self,
                                     images_dirpath=self.params.image_dataset_path)

        return det_df, lanes_dict

    def set_adapter(self, cametra_data_path: str):
        self.adapter = LanesModelAdapter(
            cametra_data_path=cametra_data_path)

    def set_distance_from_road_boundary_threshold(self, distance_from_road_boundary_threshold: float):
        self.params.distance_from_road_boundary_threshold = distance_from_road_boundary_threshold

    def apply(self,
              detections: pd.DataFrame,
              save_path=None) -> Tuple[pd.DataFrame, dict]:

        sub_results_df_arr = []
        lanes_dict = {
            "frames": {}
        }
        failed_frame_ids = []

        for frame_name in detections.name.unique():
            frame_detections = detections[detections.name == frame_name].copy()
            frame_detections.loc[:, 'is_outside_boundaries'] = BoundariesLabel.NOT_DECIDED.name
            frame_detections.loc[:, 'lane_assignment'] = RelativeLanePosition.NOT_DECIDED.name
            frame_detections.loc[:, 'is_above_horizon'] = False
            frame_detections.loc[:, 'found_both_boundaries'] = False

            try:
                sub_results_df, sub_lanes_dict = self._apply_single(
                    frame_detections=frame_detections)

                # The next logic should be fixed universaly
                name = str(frame_name)
                name = name[:-4] if '.png' in name else name
                lanes_dict['frames'][name] = sub_lanes_dict

            except NoCametraDataException as e:
                logger.warning(
                    f'[{frame_name}] {e}. Skipping frame.')

                failed_frame_ids.append(frame_name)
                sub_results_df = frame_detections

            except Exception as e:
                logger.error(
                    f'[{frame_name}] {traceback.format_exc()}')

                failed_frame_ids.append(frame_name)
                sub_results_df = frame_detections

            sub_results_df_arr.append(sub_results_df)

        if len(sub_results_df_arr) > 0:
            df_with_results = pd.concat(sub_results_df_arr) if len(
                sub_results_df_arr) > 0 else pd.DataFrame(columns=detections.columns)
        else:
            df_with_results = detections.copy()

        total_frames = len(detections.name.unique())
        total_fail = len(failed_frame_ids)

        if total_fail > 0:
            logger.debug(
                f'Failed for {total_fail} frames out of {total_frames} frames. Failed frames are {failed_frame_ids}')
        else:
            logger.debug(
                f'Success for all {total_frames} frames.')

        if save_path:
            if not os.path.exists(Path(save_path)):
                os.makedirs(Path(save_path))
            df_with_results.to_csv(
                Path(save_path).joinpath('results_df_with_rbs.tsv'), sep='\t', index=False)

            # Save lanes dict as json
            with open(Path(save_path).joinpath('lanes.json'), 'w') as f:
                json.dump(lanes_dict, f)

        if self.params.save_plots:
            logger.info(f'Saved plots to {self.params.save_plots}')

        return df_with_results, lanes_dict
