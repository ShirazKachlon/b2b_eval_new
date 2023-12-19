from pathlib import Path

evaluation_gt_columns = ['name', 'x_center', 'y_center', 'width', 'height', 'label', 'd3_separation', 'r_label',
                         'l_label']
evaluation_detection_columns = evaluation_gt_columns.copy()
evaluation_detection_columns.append('score')

config_template_path = Path.cwd() / 'src' / 'eval' / 'configs' / 'eval_wrapper_config_8mp.json'
tmp_folder_path = Path.cwd() / 'tmp'