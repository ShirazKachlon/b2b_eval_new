from src.eval.utils import load_tsv_to_df, save_df_to_tsv


class DataPreparations:

    def convert_tsv_template(self, filter_func, input_file_path, extension: str = '_parsed.tsv'):
        output_file_path = input_file_path.replace('.tsv', extension)
        df = load_tsv_to_df(input_file_path)
        gt_df = filter_func(df)
        save_df_to_tsv(gt_df, output_file_path)
        return output_file_path

    def filter_objects_out_of_corridor(self, df):
        filtered_df = ObjectDistanceEstimator(df)
        return filtered_df

    def transform_multi_frame_bounding_box_from_3d_to_2d(self, df):
        transformed_df = convert_3d_2d(df)
        return transformed_df

    def remove_columns(self, df, columns_to_keep):
        filterd_df = df[columns_to_keep]
        return filterd_df
