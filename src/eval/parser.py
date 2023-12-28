import os
import shutil

from src.eval.consts import evaluation_gt_columns, evaluation_detection_columns
from src.eval.utils import load_tsv_to_df, save_df_to_tsv
from src.eval.services.correct_2d_by_3d import correct_2d_by_3d


class Parser(object):
    def __init__(self):
        self.config = None
        self.gt_df = None
        self.det_df = None
        self.det_path = None
        self.gt_path = None
        self.dir_for_save = None
        self.origin_file_dir = None

    def set_config(self, config, dir_for_save):
        self.dir_for_save = dir_for_save
        self.origin_file_dir = os.path.join(dir_for_save, 'original_files')
        os.makedirs(self.origin_file_dir, exist_ok=True)
        self.config = config

        self.gt_path = config['ground_truth_path']
        self.det_path = config['det_path']

        shutil.copyfile(self.gt_path, os.path.join(self.origin_file_dir, os.path.basename(self.gt_path)))
        shutil.copyfile(self.det_path, os.path.join(self.origin_file_dir, os.path.basename(self.det_path)))

        self.gt_df = load_tsv_to_df(self.gt_path)
        self.det_df = load_tsv_to_df(self.det_path)

    def update_config(self):
        self.config['ground_truth_path'] = self.gt_path
        self.config['det_path'] = self.det_path
        return self.config

    def parse(self, config, dir_for_save):
        self.set_config(config, dir_for_save)
        self.parse_auto_labeling()
        self.parse_multi_frame()
        self.parse_lanes_filter()
        self.save()

        return self.config

    def parse_auto_labeling(self):
        if self.config['is_autolabeling_gt']:
            self.gt_path = self.gt_path.replace('.tsv', '_as_gt.tsv')
            self.gt_path = os.path.join(self.dir_for_save, os.path.basename(self.gt_path))
            self.gt_df = self.gt_df[evaluation_gt_columns]
            self.gt_df = self.update_columns_for_eval_v2(self.gt_df)
        else:
            self.det_path = self.det_path.replace('.tsv', '_parsed.tsv')
            self.det_path = os.path.join(self.dir_for_save, os.path.basename(self.det_path))
            self.det_df = self.det_df[evaluation_detection_columns]
        return self.update_config()

    def parse_multi_frame(self):
        if not self.config['is_multi_frame_detection']:
            return
        # TODO: occluded MF correction
        self.det_df = correct_2d_by_3d(self.det_df)
        self.det_path = self.det_path.replace('.tsv', '_with_truncated_2d_boxes.tsv')
        self.det_path = os.path.join(self.dir_for_save,os.path.basename(self.det_path))
        self.update_config()

    def parse_lanes_filter(self):
        if not self.config['is_lanes_filter']:
            return
        # TODO: Add logic for lanes filter
        # self.det_df = lanes_filter(self.det_df, self.config['cametra_path'], self.config['imu_path'])
        self.det_path = self.det_path.replace('.tsv', '_with_lanes_filter.tsv')
        self.update_config()

    def update_columns_for_eval_v2(self, df):
        columns = {'is_occluded': 'is_occluded_gt', 'is_truncated':'is_truncated_gt'}
        df.rename(columns=columns, inplace=True)
        return df

    def save(self):
        copy_gt_path = os.path.join(self.dir_for_save, os.path.basename(self.gt_path))
        save_df_to_tsv(self.gt_df, copy_gt_path)

        copy_det_path = os.path.join(self.dir_for_save, os.path.basename(self.det_path))
        save_df_to_tsv(self.det_df, copy_det_path)
