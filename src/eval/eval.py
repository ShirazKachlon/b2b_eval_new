import os
from copy import copy
from pathlib import Path

from automotive.evaluation.evaluate_multiple_classes import eval_multi_classes

from src.eval.consts import config_template_path, Inputs, tmp_config_path
from src.eval.pre_process_data import DataPreparations
# from src.eval.consts import summary_dict_for_eval
from src.eval.utils import load_json, dump_json


class Evaluation:
    def __init__(self, config_path=None):
        self.parser = DataPreparations()
        self.config_path = config_path
        self.list_of_summarize_dict = None

    def run_evaluation(self, **kwargs):
        #bpre_proccess_data(flags -> easy money)
        if self.config_path is None:
            self.build_config(**kwargs)

        self.validate_config()
        eval_multi_classes(self.config_path)
        self.cleanup()

    def build_config(self, **kwargs):
        cur_config = load_json(config_template_path)
        cur_config = self.update_config(cur_config, **kwargs)
        dump_json(tmp_config_path, cur_config)

    def cleanup(self):
        if tmp_config_path.is_file():
            Path.unlink(tmp_config_path)

    def update_config(self, eval_config, **kwargs):
        cur_config = copy(eval_config)

        for cur_input in Inputs.get_all_inputs():
            if cur_input not in kwargs:
                raise ValueError(f'{cur_input} is missing from inputs')
            cur_config[cur_input] = kwargs.get(cur_input)

        for i, config in enumerate(cur_config['configs']):
            crop_path = config['evaluation_configuration']
            cur_config['configs'][i]['evaluation_configuration'] = Path.cwd() / crop_path

        return cur_config

    def validate_config(self):
        # TODO: create a config validator
        pass

# Evaluation(r'/home/shiraz/PycharmProjects/b2b_eval_new/src/eval/tests/configs/eval_wrapper_config_8mp.json').run_evaluation()
