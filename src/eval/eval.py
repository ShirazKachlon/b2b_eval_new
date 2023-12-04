import os
from copy import copy
from pathlib import Path

from automotive.evaluation.evaluate_multiple_classes import eval_multi_classes

from src.eval.config_handler import ConfigHandler
from src.eval.consts import config_template_path, tmp_folder_path, evaluation_gt_columns, evaluation_detection_columns
from src.eval.data_prepartions import DataPreparations
from src.eval.utils import load_json, dump_json, cleanup_directory, validate_input, validate_path_input


class Evaluation:
    def __init__(self, config_path=None):
        self.parser = DataPreparations()
        self.config_path = config_path
        self.cur_tmp_folder_path = tmp_folder_path
        self.list_of_summarize_dict = None

    def run_evaluation(self, **kwargs):
        # bpre_proccess_data(flags -> easy money)
        if self.config_path is None:
            self.build_eval_for_running(**kwargs)

        eval_multi_classes(str(self.cur_tmp_folder_path / 'config.json'))
        cleanup_directory(self.cur_tmp_folder_path / 'config.json')

    def build_eval_for_running(self, **kwargs):
        os.makedirs(self.cur_tmp_folder_path, exist_ok=True)
        cur_config = load_json(config_template_path)
        cur_config = self.update_config(cur_config, **kwargs)
        dump_json(self.cur_tmp_folder_path / 'config.json', cur_config)

    def update_config(self, eval_config, **kwargs):
        cur_config = copy(eval_config)
        for cur_input in ConfigHandler.get_all_inputs():
            if cur_input not in kwargs:
                raise ValueError(f'{cur_input} is missing from inputs')
            cur_config[cur_input] = kwargs.get(cur_input)
        cur_config[cur_input] = kwargs.get(cur_input)

        if cur_config['is_autolabeling_gt']:
            cur_config['ground_truth_path'] = self.parser.remove_columns(evaluation_gt_columns,
                                                                         cur_config['ground_truth_path'],
                                                                         '_as_gt.tsv')
        else:
            cur_config['det_path'] = self.parser.remove_columns(evaluation_detection_columns, cur_config['det_path'])

        # TODO: make it in a better way
        if cur_config['is_multi_frame_detection']:
            cur_config['det_path'] = self.parser.convert_tsv_template(
                self.parser.transform_multi_frame_bounding_box_from_3d_to_2d, cur_config['det_path'], '_2d_bbox.tsv')
        if cur_config['is_corridor_filter']:
            cur_config['ground_truth_path'] = self.parser.convert_tsv_template(
                self.parser.filter_objects_out_of_corridor, cur_config['ground_truth_path'], '_corridor_filtered.tsv')
            cur_config['det_path'] = self.parser.convert_tsv_template(self.parser.filter_objects_out_of_corridor,
                                                                      cur_config['det_path'], '_corridor_filtered.tsv')
        if cur_config['is_lanes_filter']:
            cur_config['ground_truth_path'] = self.parser.convert_tsv_template(
                self.parser.filter_objects_out_by_lane_filter, cur_config['ground_truth_path'], '_lanes_filtered.tsv')
            cur_config['det_path'] = self.parser.convert_tsv_template(self.parser.filter_objects_out_by_lane_filter,
                                                                      cur_config['det_path'], '_lanes_filtered.tsv')

        for i, config in enumerate(cur_config['configs']):
            crop_path = config['evaluation_configuration']
            cur_config['configs'][i]['evaluation_configuration'] = str(Path.cwd() / crop_path)
        self.validate_config(cur_config)

        return cur_config

    def validate_config(self, config):
        config_handler = ConfigHandler
        inputs_to_check = config_handler.get_inputs_type()

        # Check lane filter inputs
        if config['is_lanes_filter'] and (config['cametra_path'] is None or config['imu_path'] is None):
            raise ValueError(f'lanes filter require cametra path and imu path')
        else:
            inputs_to_check.pop(config_handler.cametra_path, None)
            inputs_to_check.pop(config_handler.imu_path, None)

        # Check other inputs
        for input_parm in inputs_to_check:
            if input_parm in config_handler.get_path_inputs():
                validate_path_input(input_parm, config[input_parm])
            else:
                validate_input(input_parm, config[input_parm], config_handler.get_inputs_type()[input_parm])
