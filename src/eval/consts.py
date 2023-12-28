from pathlib import Path

evaluation_gt_columns = ['name', 'x_center', 'y_center', 'width', 'height', 'label', 'd3_separation', 'r_label',
                         'l_label','is_occluded', 'is_truncated']
evaluation_detection_columns = evaluation_gt_columns.copy()
evaluation_detection_columns.append('score')

config_template_path = Path.cwd() / 'src' / 'eval' / 'configs' / 'eval_wrapper_config_8mp.json'
tmp_folder_path = Path.cwd() / 'tmp'

bins_to_range = {
    '327-1920': '0-10',
    '165-327': '10-20',
    '110-165': '20-30',
    '82-110': '30-40',
    '66-82': '40-50',
    '55-66': '50-60',
    '47-55': '60-70',
    '41-47': '70-80',
    '37-41': '80-90',
    '33-37': '90-100',
    '192-1920': '0-10',
    '92-192': '10-20',
    '61-92': '20-30',
    '45-61': '30-40',
    '37-45': '40-50',
    '31-37': '50-60',
    '26-31': '60-70',
    '23-26': '70-80',
    '20-23': '80-90',
    '18-20': '90-100',
    '273-1920': '0-10',
    '137-273': '10-20',
    '91-137': '20-30',
    '68-91': '30-40',
    '55-68': '40-50',
    '46-55': '50-60',
    '39-46': '60-70',
    '34-39': '70-80',
    '30-34': '80-90',
    '27-30': '90-100',
    '22-27': '100-125',
    '18-22': '125-150',
    '14-18': '150-200',
    '11-14': '200-250'
}
