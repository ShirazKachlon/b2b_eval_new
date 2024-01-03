import os
import shutil
from copy import copy
from pathlib import Path

from automotive.evaluation.evaluate_multiple_classes import eval_multi_classes
from automotive.evaluation_v2.run_od_evaluation import run_eval_v2
from src.eval.config_handler import ConfigHandler
from src.eval.consts import config_template_path, tmp_folder_path
from src.eval.labels import labels_if_autolabeling_gt, labels_if_autolabeling_det
from src.eval.output_organizer import OutputOrganizer
from src.eval.parser import Parser
from src.eval.post_process import PostProcess
from src.eval.utils import load_json, dump_json, validate_input, validate_path_input


class Evaluation:
    def __init__(self, config_path=None, eval_version=None):
        self.config_path = config_path
        self.eval_version = eval_version
        self.cur_tmp_folder_path = tmp_folder_path
        self.list_of_summarize_dict = None
        self.main_output_dir = None
        self.parser = Parser()

    def run_evaluation(self, **kwargs):
        tmp_config_path = self.build_eval_for_running(**kwargs)
        if self.eval_version == 1:
            "Eval version 1"
            eval_multi_classes(tmp_config_path)
        else:
            "Eval version 2"
            run_eval_v2(tmp_config_path)

        PostProcess(self.eval_version, self.main_output_dir.output_folder(),
                    self.main_output_dir.summary_folder()).run()

    def build_eval_for_running(self, **kwargs):
        update_needed = False

        if self.config_path is None:
            cur_config = load_json(config_template_path)
            cur_config = self.update_args_in_template_config(cur_config, **kwargs)
            update_needed = True
        else:
            cur_config = load_json(self.config_path)

        self.main_output_dir = OutputOrganizer(cur_config['output_dir'])
        cur_config['output_dir'] = self.main_output_dir.output_folder()
        output_input_dir = self.main_output_dir.input_folder()
        cur_crops_dir = self.main_output_dir.crops_folder()
        if os.path.exists(cur_crops_dir):
            # If it exists, remove it before copying
            shutil.rmtree(cur_crops_dir)
        # TODO: fix bug of copying wrong crops from repo instead of the config
        shutil.copytree(os.path.join(Path.cwd(), 'src', 'eval', 'configs', 'crops'), cur_crops_dir)

        if update_needed:
            cur_config = self.update_crops_in_config(cur_config)

        self.validate_config(cur_config)
        cur_config = self.parser.parse(cur_config, output_input_dir)
        self.config_path = os.path.join(output_input_dir, 'config.json')
        dump_json(self.config_path, cur_config)
        return str(self.config_path)

    @staticmethod
    def update_args_in_template_config(eval_config, **kwargs):
        cur_config = copy(eval_config)
        for cur_input in ConfigHandler.get_all_inputs():
            if cur_input not in kwargs:
                raise ValueError(f'{cur_input} is missing from inputs')
            cur_config[cur_input] = kwargs.get(cur_input)
        cur_config[cur_input] = kwargs.get(cur_input)

        return cur_config

    def update_crops_in_config(self, eval_config):
        eval_labels = labels_if_autolabeling_gt if eval_config['is_autolabeling_gt'] else labels_if_autolabeling_det
        for i, config in enumerate(eval_config['configs']):
            crop_path = config['evaluation_configuration']
            absolute_crop_path = os.path.join(self.main_output_dir.crops_folder(), os.path.basename(crop_path))
            eval_config['configs'][i]['evaluation_configuration'] = str(absolute_crop_path)
            cur_class_config = load_json(absolute_crop_path)
            for cur_update in eval_labels[config['config_name']]:
                cur_class_config['data_config'][cur_update] = eval_labels[config['config_name']][cur_update]
            dump_json(absolute_crop_path, cur_class_config)
        return eval_config

    @staticmethod
    def validate_config(config):
        config_handler = ConfigHandler
        inputs_to_check = config_handler.get_inputs_type()

        # Check lane filter inputs
        if config['is_lanes_filter'] and (config['cametra_folder_path'] is None):
            raise ValueError(f'lanes filter require cametra folser path')
        else:
            inputs_to_check.pop(config_handler.cametra_path, None)
            inputs_to_check.pop(config_handler.imu_path, None)

        # Check other inputs
        for input_parm in inputs_to_check:
            if input_parm in config_handler.get_path_inputs():
                validate_path_input(input_parm, config[input_parm])
            else:
                validate_input(input_parm, config[input_parm], config_handler.get_inputs_type()[input_parm])

    @staticmethod
    def evaluation_version(config_path):
        config = load_json(config_path)
        if 'eval_version' in config.keys():
            return config['eval_version']
        else:
            return -1
