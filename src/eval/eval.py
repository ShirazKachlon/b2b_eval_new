from automotive.evaluation.evaluate_multiple_classes import eval_multi_classes

from src.eval import parser
from src.eval.consts import summary_dict_for_eval
from src.eval.utils import load_json, dump_json, get_user


class Evaluation:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config_template_path = None
        self.inputs = None
        self.list_of_summarize_dict = None

    def run_evaluation(self):
        eval_multi_classes(self.config_path)

    def build_config(self, **kwargs):
        pass
        # cur_config = load_json(self.config_template_path)
        # self.update_config(cur_config)
        # dump_json(self.config_path, cur_config)

    def update_config(self, config):
        config['output_dir'] = self.inputs['evaluation_dir_path']
        config['ground_truth_path'] = self.inputs['ground_truth_file_path']
        config['det_path'] = self.inputs['detections_file_path']
        config['images_path'] = self.inputs['images_dir_path']
        config['save_results'] = self.inputs['save_results']
        for i, config in enumerate(config['configs']):
            path1 = config['evaluation_configuration']
            config['configs'][i]['evaluation_configuration'] = path1.replace('{user}', get_user())

        return config


# Evaluation(r'/home/shiraz/PycharmProjects/b2b_eval_new/src/eval/tests/configs/eval_wrapper_config_8mp.json').run_evaluation()