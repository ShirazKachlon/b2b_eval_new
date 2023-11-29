from pathlib import Path


class Inputs:
    output_dir = 'output_dir'
    ground_truth_path = 'ground_truth_path'
    det_path = 'det_path'
    save_results = 'save_results'
    images_path = 'images_path'
    autolabeling_is_gt = 'autolabeling_is_gt'
    multi_frame_detection = 'multi_frame_detection'

    @classmethod
    def get_all_inputs(cls):
        return [cls.output_dir, cls.ground_truth_path, cls.det_path, cls.save_results, cls.images_path,
                cls.autolabeling_is_gt, cls.multi_frame_detection]

evaluation_gt_columns = ['name', 'x_center', 'y_center', 'width', 'height', 'label', 'd3_separation', 'r_label',
                         'l_label', 'score']
evaluation_detection_columns = ['name', 'x_center', 'y_center', 'width', 'height', 'label', 'd3_separation', 'score']

config_template_path = Path.cwd() / 'src' / 'eval' / 'configs' / 'eval_wrapper_config_8mp.json'
tmp_config_path =Path.cwd() / 'src' / 'eval' / 'configs' / 'tmp_config.json'