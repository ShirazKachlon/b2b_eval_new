import json
from pathlib import Path

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


def cleanup_directory(path):
    if path.is_dir():
        for item in path.iterdir():
            cleanup_directory(item)
        path.rmdir()
    else:
        path.unlink()


def validate_input(parm_name, parm, parm_type):
    if not isinstance(parm, parm_type):
        raise ValueError(f'{parm_name} must be {parm_type}')


def validate_path_input(parm_name, parm):
    if not (isinstance(parm, (Path, str)) and Path(parm).exists()):
        raise ValueError(f'{parm_name} must be a valid path')
