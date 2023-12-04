from src.eval.utils import load_tsv_to_df, save_df_to_tsv


class DataPreparations:

    def remove_columns(self, columns_to_keep, input_file_path, extension: str = '_parsed.tsv'):
        output_file_path = input_file_path.replace('.tsv', extension)
        df = load_tsv_to_df(input_file_path)
        filterd_df = df[columns_to_keep]
        save_df_to_tsv(filterd_df, output_file_path)
        return output_file_path

    def convert_tsv_template(self, filter_func, input_file_path, extension: str = '_parsed.tsv'):
        output_file_path = input_file_path.replace('.tsv', extension)
        df = load_tsv_to_df(input_file_path)
        gt_df = filter_func(df)
        save_df_to_tsv(gt_df, output_file_path)
        return output_file_path

    def filter_objects_out_of_corridor(self, df):
        # filtered_df = ObjectDistanceEstimator(df)
        filtered_df = df
        return filtered_df

    def filter_objects_out_by_lane_filter(self, df):
        filtered_df = df
        return filtered_df

    def transform_multi_frame_bounding_box_from_3d_to_2d(self, df):
        # transformed_df = convert_3d_2d(df)
        transformed_df = df
        return transformed_df

# actions = {
#     'is_autolabeling_gt': {
#         'col_to_keep': evaluation_gt_columns,
#         'extension': '_as_gt.tsv'
#     },
#     'is_multi_frame_detection': {
#         'function': self.parser.transform_multi_frame_bounding_box_from_3d_to_2d,
#         'extension': '_2d_bbox.tsv'
#     },
#     'is_corridor_filter': {
#         'function': self.parser.filter_objects_out_of_corridor,
#         'extension': '_corridor_filtered.tsv'
#     },
#     'is_lanes_filter': {
#         'function': self.parser.filter_objects_out_by_lane_filter,
#         'extension': '_lanes_filtered.tsv'
#     }
# }
