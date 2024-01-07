from automotive.aeb.src.camera_model import CameraModel
import sys
import json
import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial
from typing import List
import logging
from functools import lru_cache

logger: logging.Logger = logging.getLogger("lanes")


class NoCametraDataException(Exception):
    pass


class Lane:

    def __init__(self,
                 boundary_name: str,
                 points_xs: List[float],
                 points_ys: List[float]):
        """
        Lane class. Fit deg-3 polynomial on given points.

        Args:
            boundary_name (str): the name of the lane.
            points_xs (List[float]): initial xs points to fit on.
            points_ys (List[float]): initial ys points to fit on.
        """

        self.boundary_name = boundary_name
        self.points_xs = np.array(points_xs)
        self.points_ys = np.array(points_ys)

        self.poly = Polynomial(np.polyfit(
            self.points_ys, self.points_xs, deg=3)[::-1])

        self.active_region = (self.points_ys.min(), self.points_ys.max())

    def __call__(self, ys: List[float]) -> List[float]:
        return self.poly(ys)

    def get_coefs(self) -> np.ndarray:
        return self.poly.coef

    def get_inside_image_mask(self):
        valid_xs = np.logical_and(self.points_xs >= 0, self.points_xs <= 3840)
        valid_ys = np.logical_and(self.points_ys >= 0, self.points_ys <= 1920)
        return np.logical_and(valid_xs, valid_ys)

    def get_distance(self, xs, ys):
        dists = [np.linalg.norm(np.array([x - self.points_xs, y - self.points_ys]), axis=0).min()
                 for x, y in zip(xs, ys)]
        return np.array(dists)


class Lanes:

    def __init__(self, frame_df: pd.DataFrame, camera_model: CameraModel):
        """
        Lanes class, given frame_df which contains polynomial coefficients for existing lanes, creates relevant lanes.

        Args:
            frame_df (pd.DataFrame): DataFrame contains the existing found lanes.
            camera_model (CameraModel): the associated camera model for the given frame.
        """

        self.ego_right: Lane = None
        self.ego_left: Lane = None
        self.adjacent_right: Lane = None
        self.adjacent_left: Lane = None
        self.road_boundary_right: Lane = None
        self.road_boundary_left: Lane = None

        for _, lane_df in frame_df.iterrows():
            coefs = lane_df[['polynomial[0]', 'polynomial[1]',
                             'polynomial[2]', 'polynomial[3]']].values
            poly = Polynomial(coef=coefs)
            points = [(poly(z), 0, z) for z in np.arange(0, 50, 0.5)]
            projected_points = np.array(
                [camera_model.transform_world_to_image(p) for p in points])

            lane = Lane(boundary_name=lane_df.boundary_name,
                        points_xs=projected_points[:, 0],
                        points_ys=projected_points[:, 1])
            if lane.get_inside_image_mask().sum() < 20:
                # There are not enough points inside the image for this lane
                continue
            self.__setattr__(lane_df.boundary_name, lane)

        self.length = len(frame_df)

    def __iter__(self):
        return iter(l for l in [self.road_boundary_left,
                                self.adjacent_left,
                                self.ego_left,
                                self.ego_right,
                                self.adjacent_right,
                                self.road_boundary_right] if l is not None)

    def __len__(self):
        return self.length

    def to_dict(self):
        # Creates a json from the lanes object
        lanes_dict = {}
        for lane in self:
            lanes_dict[lane.boundary_name] = {
                'poly_coefs': lane.get_coefs().tolist(),
                'min_y': lane.active_region[0],
                'max_y': lane.active_region[1],
                'points_xs': lane.points_xs.tolist(),
                'points_ys': lane.points_ys.tolist()
            }
        return lanes_dict

    def has_both_boundaries(self):
        return self.road_boundary_left is not None and self.road_boundary_right is not None


class LanesModelAdapter:
    def __init__(self, cametra_data_path: str):
        """
        LanesModelAdapter translate the existing multi-frame lanes model data to Lanes class.

        Args:
            cametra_data_path (str): Path for cametra data.
        """

        self.cametra_data_path = cametra_data_path

        self.calib_data_path = f'{cametra_data_path}/imu_data.json'
        self.imu_data_path = f'{cametra_data_path}/imu_data.csv'
        self.cametra_output_df_path = f'{cametra_data_path}/cametra_interface_lanes_output.tsv'

        self.calib_dict = json.load(open(self.calib_data_path, 'r'))
        self.imu_df = pd.read_csv(self.imu_data_path)
        self.cametra_output_df = pd.read_csv(
            self.cametra_output_df_path, sep='\t')

        logger.info(
            f'Preparing adapter for f{cametra_data_path} of size {self.cametra_output_df.shape[0]}, this might take a few moments ..')

        height, focal_length, cx, cy, overhang_front = self._get_camera_initial_parameters(
            self.calib_dict, self.imu_df)
        self.camera_model = CameraModel(height=height,
                                        focus_distance=focal_length,
                                        cx=cx,
                                        cy=cy,
                                        fixed_dims={},
                                        front_overhang=overhang_front,
                                        highway_speed_threshold_kmh=65,
                                        transform_mode="ROAD_CONTACT_POINT")

        self.imu_df = self._update_imu_df(self.imu_df, self.calib_dict)

    def _get_camera_initial_parameters(self, calib_dict, imu_df):
        first_frame = imu_df.loc[0, 'name']
        focal_length, cx, cy = imu_df.loc[0,
                                          'focal_length'], imu_df.loc[0, 'ppx'], imu_df.loc[0, 'ppy']
        height, overhang_front = calib_dict[str(first_frame)]['initial']['cam_pose_tz'], calib_dict[str(
            first_frame)]['initial']['overhang_front']

        return height, focal_length, cx, cy, overhang_front

    def _update_imu_df(self, imu_df, calib_dict):
        updated_df = imu_df.copy()

        for i, row in imu_df.iterrows():
            updated_df.loc[i, 'cam_pose_tx'] = calib_dict[str(
                int(row.timestamp))]['dynamic']['cam_pose_tx']
            updated_df.loc[i, 'cam_pose_ty'] = calib_dict[str(
                int(row.timestamp))]['dynamic']['cam_pose_ty']
            updated_df.loc[i, 'cam_pose_tz'] = calib_dict[str(
                int(row.timestamp))]['dynamic']['cam_pose_tz']

            updated_df.loc[i, 'yaw'] = calib_dict[str(
                int(row.timestamp))]['dynamic']['yaw']
            updated_df.loc[i, 'roll'] = calib_dict[str(
                int(row.timestamp))]['dynamic']['roll']
            updated_df.loc[i, 'pitch'] = calib_dict[str(
                int(row.timestamp))]['dynamic']['pitch']

        return updated_df

    @lru_cache(maxsize=None)
    def get_lanes(self, frame_id: int) -> Lanes:
        if frame_id not in self.imu_df.name.values:
            raise NoCametraDataException(
                f'Could not find {frame_id} in imu dataframe.')

        if frame_id not in self.cametra_output_df.frame_id.values:
            raise NoCametraDataException(
                f'Could not find {frame_id} in camertra dataframe.')

        idx = self.imu_df[self.imu_df.name == frame_id].index[0]
        self.camera_model.update_extrinsics(
            self.imu_df.loc[idx, :], horizon=None)

        frame_df = self.cametra_output_df[self.cametra_output_df.frame_id == frame_id]

        lanes = Lanes(frame_df=frame_df, camera_model=self.camera_model)
        return lanes

    def get_horizon(self, frame_id: int) -> Polynomial:
        idx = self.imu_df[self.imu_df.name == frame_id].index[0]
        pitch, roll = self.imu_df.iloc[idx][['pitch', 'roll']].values
        a, b = self.camera_model.get_horizon_from_pitch_and_roll(
            pitch=pitch, roll=roll)
        return Polynomial(coef=[b, a])
