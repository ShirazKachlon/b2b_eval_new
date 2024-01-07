import os

import pandas as pd

from automotive.evaluation.tools.nets_comparison.evaluation_log_parser import EvaluationLogParser
from src.eval.consts import bins_to_range, cols_summary_table_eval_v1, cols_summary_table_eval_v2
from src.eval.utils import load_parquet_to_df, save_df_to_tsv


class PostProcess:
    def __init__(self, eval_version, output_dir, summary_dir):
        self.eval_version = eval_version
        self.output_dir = output_dir
        self.summary_dir = summary_dir
        self.final_results = dict()
        self.df_summary_table = pd.DataFrame()

    def run(self):
        self.collect_results()
        self.merge_results()
        self.refactoring_table()
        save_df_to_tsv(self.df_summary_table, os.path.join(self.summary_dir, 'summary_table_score_0.tsv'))

    def collect_results(self):
        if self.eval_version == 1:
            self.output_dir = os.path.join(self.output_dir, 'output')
        dirs = [d for d in os.listdir(self.output_dir) if os.path.isdir(os.path.join(self.output_dir, d))]
        for d in dirs:
            for sub_dir in os.listdir(os.path.join(self.output_dir, d)):
                if sub_dir.startswith('kpi') or sub_dir.endswith('_log.log'):
                    self.final_results[d] = os.path.join(self.output_dir, d, sub_dir)
                    break
        return self.final_results

    def prepare_df_eval_v2(self, file):
        cur_df = load_parquet_to_df(self.final_results[file])
        cur_df['class'] = file
        return cur_df

    def prepare_df_eval_v1(self, file):
        evaluation_log_obj = EvaluationLogParser(self.final_results[file]).get_evaluation_log_obj()
        rows_to_append = []
        for bin_height in evaluation_log_obj:
            new_row = {
                'class': file,
                'min_height': bin_height['bin_min_height'],
                'max_height': bin_height['bin_max_height'],
                'samples': bin_height['0']['recall'].split('/')[1].split(' ')[0],
                'recall': float(bin_height['0']['recall'].split('=')[1]),
                'precision_loose': bin_height['0']['precision_loose'],
                'precision_strict': bin_height['0']['precision_strict'],
                'fa': bin_height['0']['fa'],
                'fa_localization': bin_height['0']['fa_localization'],
                'fa_random': bin_height['0']['fa_random'],
                'fppi': bin_height['0']['fppi'],
                'tp': bin_height['0']['true_positives'],
                'avg_iou': bin_height['0']['avg iou']
            }
            rows_to_append.append(new_row)
        cur_df = pd.DataFrame(rows_to_append)
        return cur_df

    def merge_results(self):
        sub_dfs = []
        for file in self.final_results:
            if self.eval_version == 2:
                cur_df = self.prepare_df_eval_v2(file)
            else:
                cur_df = self.prepare_df_eval_v1(file)
            sub_dfs.append(cur_df)
        self.df_summary_table = pd.concat(sub_dfs)
        return self.df_summary_table

    def refactoring_table(self):
        self.df_summary_table.rename(columns={'gt': 'samples'}, inplace=True)
        self.df_summary_table = self.df_summary_table.drop(
            self.df_summary_table[
                (self.df_summary_table['min_height'] == -1) | (self.df_summary_table['max_height'] == "-1")].index)
        self.df_summary_table = self.df_summary_table.drop(
            self.df_summary_table[(self.df_summary_table['min_height']) == (self.df_summary_table['max_height'])].index)
        self.df_summary_table['height(pixels)'] = (
            self.df_summary_table['min_height'].astype(int).astype(str) + "-" +
            self.df_summary_table['max_height'].astype(int).astype(str)
        )
        self.df_summary_table['ranges(m)'] = self.df_summary_table['height(pixels)'].apply(
            lambda x: bins_to_range.get(x, 'Unknown'))
        if self.eval_version == 2:
            self.df_summary_table['precision'] = (self.df_summary_table['tp'] / (
                self.df_summary_table['tp'] + self.df_summary_table['fa'])) * 100
        self.df_summary_table = self.df_summary_table[cols_summary_table_eval_v1 if self.eval_version == 1 else
                                                      cols_summary_table_eval_v2]
        return self.df_summary_table
