from pathlib import Path


# TODO: maybe move to Pydantic
class ConfigHandler:
    output_dir = 'output_dir'
    ground_truth_path = 'ground_truth_path'
    det_path = 'det_path'
    save_results = 'save_results'
    is_autolabeling_gt = 'is_autolabeling_gt'
    is_multi_frame_detection = 'is_multi_frame_detection'
    is_lanes_filter = 'is_lanes_filter'
    cametra_path = 'cametra_path'
    imu_path = 'imu_path'
    evaluate_single_score = 'evaluate_single_score'

    @classmethod
    def get_all_inputs(cls):
        return [cls.output_dir, cls.ground_truth_path, cls.det_path, cls.save_results,
                cls.is_autolabeling_gt, cls.is_multi_frame_detection, cls.is_lanes_filter,
                cls.cametra_path, cls.imu_path, cls.evaluate_single_score]

    @classmethod
    def get_inputs_type(cls):
        return {
            cls.output_dir: Path,
            cls.ground_truth_path: Path,
            cls.det_path: Path,
            cls.save_results: str,
            cls.is_autolabeling_gt: bool,
            cls.is_multi_frame_detection: bool,
            cls.is_lanes_filter: bool,
            cls.cametra_path: Path,
            cls.imu_path: Path,
            cls.evaluate_single_score: int
        }

    @classmethod
    def get_path_inputs(cls):
        return [cls.ground_truth_path, cls.det_path, cls.output_dir]
