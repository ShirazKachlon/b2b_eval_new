from src.eval.consts import evaluation_gt_columns, evaluation_detection_columns
from src.eval.filters.corridor_filter import corridor_filter
from src.eval.filters.lanes_filter import lanes_filter
from src.eval.filters.mf_parser import convert_3d_2d
from src.eval.utils import load_tsv_to_df, save_df_to_tsv


class Parser(object):
    def __init__(self):
        self.config = None
        self.gt_df = None
        self.det_df = None
        self.det_path = None
        self.gt_path = None

    def set_config(self, config):
        self.config = config

        self.gt_path = config['ground_truth_path']
        self.det_path = config['det_path']

        self.gt_df = load_tsv_to_df(self.gt_path)
        self.det_df = load_tsv_to_df(self.det_path)

    def parse(self, config):
        self.set_config(config)
        self.parse_auto_labeling()
        self.parse_multi_frame()
        self.parse_lanes_filter()
        self.parse_corridor_filter()
        self.save()

    def parse_auto_labeling(self):
        if self.config['is_autolabeling_gt']:
            self.gt_path = self.gt_path.replace('.tsv', '_as_gt.tsv')
            self.gt_df = self.gt_df[evaluation_gt_columns]

        else:
            self.det_path = self.det_path.replace('.tsv', '_parsed.tsv')
            self.det_df = self.det_df[evaluation_detection_columns]

    def parse_multi_frame(self):
        if not self.config['is_multi_frame_detection']:
            return

        # self.det_df = convert_3d_2d(self.det_df)
        self.det_path = self.det_path.replace('.tsv', '_with_truncated_2d_boxes.tsv')

    def parse_lanes_filter(self):
        if not self.config['is_lanes_filter']:
            return

        # self.det_df = lanes_filter(self.det_df, self.config['cametra_path'], self.config['imu_path'])
        self.det_path = self.det_path.replace('.tsv', '_with_lanes_filter.tsv')

    def parse_corridor_filter(self):
        if not self.config['is_corridor_filter']:
            return

        # self.det_df = corridor_filter(self.det_df)
        self.det_path = self.det_path.replace('.tsv', '_with_corridor_filter.tsv')

    def save(self):
        if self.gt_path != self.config['ground_truth_path']:
            save_df_to_tsv(self.gt_df, self.gt_path)

        if self.det_path != self.config['det_path']:
            save_df_to_tsv(self.det_df, self.det_path)
