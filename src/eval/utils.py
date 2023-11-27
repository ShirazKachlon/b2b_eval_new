import json
import getpass
import os

import pandas as pd


def load_json(json_filepath):
    try:
        with open(json_filepath, 'r') as json_file:
            data = json.load(json_file)
    except:
        return {}

    return data


def dump_json(json_filepath, data):
    try:
        with open(json_filepath, 'w') as json_file:
            json.dump(data, json_file, indent=2)
    except:
        pass


def save_df_to_tsv(df, output_tsv_filepath):
    df.to_csv(output_tsv_filepath, sep='\t', index=False)

def load_tsv_to_df(input_tsv_filepath):
    df = pd.read_csv(input_tsv_filepath, sep='\t')
    return df

def get_user():
    return getpass.getuser()
