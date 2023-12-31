import os

import pandas as pd

from src.eval.consts import bins_to_range
from src.eval.utils import save_df_to_tsv, load_parquet_to_df


class PostProcess:
    def __init__(self, output_dir, summary_dir):
        self.output_dir = output_dir
        self.summary_dir = summary_dir
        self.final_results = dict()
        self.df_summary_table = pd.DataFrame()

    def run(self):
        self.collect_results()
        self.merge_results()
        final_df = self.refactoring_table()
        save_df_to_tsv(final_df, os.path.join(self.summary_dir, 'summary.tsv'))

    def collect_results(self):
        dirs = [d for d in os.listdir(self.output_dir) if os.path.isdir(os.path.join(self.output_dir, d))]
        for d in dirs:
            for sub_dir in os.listdir(os.path.join(self.output_dir, d)):
                if sub_dir.startswith('kpi'):
                    self.final_results[d] = os.path.join(self.output_dir, d, sub_dir)
        return self.final_results

    def merge_results(self):
        sub_dfs = []
        for file in self.final_results:
            cur_df = load_parquet_to_df(self.final_results[file])
            cur_df.insert(0, 'class', file)
            sub_dfs.append(cur_df)
        self.df_summary_table = pd.concat(sub_dfs)
        return self.df_summary_table

    def refactoring_table(self):
        self.df_summary_table.rename(columns={'gt': 'samples'}, inplace=True)
        self.df_summary_table = self.df_summary_table.drop(
            self.df_summary_table[self.df_summary_table['min_height'] == -1].index)
        self.df_summary_table['ranges(m)'] = (
                self.df_summary_table['min_height'].astype(int).astype(str) + "-" +
                self.df_summary_table['max_height'].astype(int).astype(str)
        ).apply(lambda x: bins_to_range.get(x, 'Unknown'))
        self.df_summary_table['precision'] = (self.df_summary_table['tp'] / (
                self.df_summary_table['tp'] + self.df_summary_table['fa'])) * 100
        self.df_summary_table = self.df_summary_table[
            ['class', 'min_height', 'max_height', 'ranges(m)', 'recall', 'precision', 'fppi', 'fa', 'fa_localization',
             'fa_random',
             'tp']]
        return self.df_summary_table
