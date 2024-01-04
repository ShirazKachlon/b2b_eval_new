import argparse
import os

from src.eval.eval import Evaluation


def get_parser():
    parser = argparse.ArgumentParser(description='B2B Evaluation wrapper for company evaluation')
    subparsers = parser.add_subparsers(dest="command")

    # Parser for passing only config json file path
    file_parser = subparsers.add_parser('config_file')
    file_parser.add_argument('--config_json_path', type=str, required=True, help='Path to JSON configuration file')
    file_parser.add_argument('--eval_version', type=int, required=False, default=1, help='evaluation version')

    # Parser for using template json by adding manually inputs
    manual_parser = subparsers.add_parser('inputs')
    manual_parser.add_argument('--eval_version', type=int, required=False, default=1, help='evaluation version')
    manual_parser.add_argument('--output_dir', type=str, required=True, help='evaluation output directory')
    manual_parser.add_argument('--ground_truth_path', type=str, help='Path to ground truth file')
    manual_parser.add_argument('--det_path', type=str, help='Path to detection file')
    manual_parser.add_argument('--save_results', type=str, help='evaluation results states',
                               choices=['base', 'base_and_breakdown'])
    manual_parser.add_argument('--evaluate_single_score', type=int,
                               help='which score to use for evaluation. default 0 -> max recall.'
                                    ' use -1 to run evaluation on all scores', default=0)
    manual_parser.add_argument('--is_autolabeling_gt', action='store_true', help='flag to use autolabeling as gt')
    manual_parser.add_argument('--is_multi_frame_detection', action='store_true',
                               help='flag to use multi frame detection')
    manual_parser.add_argument('--is_lanes_filter', action='store_true',
                               help='flag to filter objects according to lanes road boundary estimation')
    manual_parser.add_argument('--cametra_path', type=str, required=False, help='Path to cametra detection file')
    manual_parser.add_argument('--imu_path', type=str, required=False, help='Path to IMU file')
    manual_parser.add_argument('--override', default=False, action='store_true',
                               help='whether to overrider old results if exists')
    manual_parser.add_argument('--sf_det_path', type=str, required=False, default='',
                               help='SF detection path for correction of occluded MF objects')

    return parser


def entry_point():
    parser = get_parser()
    args = parser.parse_args()
    args = vars(args)

    if args['override'] and os.path.exists(args['output_dir']):
        import shutil
        print(f"Overriding old results in {args['output_dir']}")
        shutil.rmtree(args['output_dir'])

    cur_eval = Evaluation(config_path=args.get('config_json_path'), eval_version=args.get('eval_version'))
    cur_eval.run_evaluation(**args)


if __name__ == '__main__':
    entry_point()
