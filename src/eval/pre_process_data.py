from src.eval.utils import load_tsv_to_df, save_df_to_tsv


class DataPreparations():
    def __init__(self):
        pass

    def filter_objects_out_of_corridor(self, input_file_path):
        output_file_path = input_file_path.replace('.tsv', '_corridor_filtered.tsv')
        df = load_tsv_to_df(input_file_path)
        filtered_df = ObjectDistanceEstimator(df)
        save_df_to_tsv(filtered_df, output_file_path)
        return output_file_path

    def transform_multi_frame_bounding_box_from_3d_to_2d(self, input_file_path):
        output_file_path = input_file_path.replace('.tsv', '_2d_bbox.tsv')
        df = load_tsv_to_df(input_file_path)
        transformed_df = convert_3d_2d(df)
        save_df_to_tsv(transformed_df, output_file_path)
        return output_file_path

    def convert_al_det_to_gt(self, input_file_path):
        output_file_path = input_file_path.replace('.tsv', '_as_gt.tsv')
        df = load_tsv_to_df(input_file_path)
        columns_to_keep = ['x_center', 'y_center', 'width', 'height', 'label', 'score', 'is_occluded', 'is_truncated', 'd3_separation', 'l_label', 'r_label', 'is_rider_on_2_wheels']
        gt_df = df[columns_to_keep]
        save_df_to_tsv(gt_df, output_file_path)
        return output_file_path
