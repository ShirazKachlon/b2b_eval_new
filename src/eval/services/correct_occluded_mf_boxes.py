import os
import tqdm
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger('evaluation')


def match_mf_to_sf_boxes_in_frame(mf_df, sf_df):
    mf_df1 = mf_df.copy()
    matching_indices = [(index1, index2) for index1, value1 in zip(mf_df.index.to_numpy(), mf_df['score'].to_numpy())
                        for index2, value2 in zip(sf_df.index.to_numpy(), sf_df['score'].to_numpy())
                        if value1 == value2]
    for idx in matching_indices:
        mf_df1.loc[idx[0], ['sf_x_center', 'sf_y_center', 'sf_width', 'sf_height']] = sf_df.loc[
            idx[1], ['x_center', 'y_center', 'width', 'height']].to_numpy()
    return mf_df1


def match_sf_to_mf_boxes(mf_df, sf_df):
    mf_df[['sf_x_center', 'sf_y_center', 'sf_width', 'sf_height']] = np.nan

    for frame, mf_frame_df in tqdm.tqdm(mf_df.groupby('name')):
        sf_frame_df = sf_df[sf_df['name'] == frame]
        mf_frame_df = match_mf_to_sf_boxes_in_frame(mf_frame_df, sf_frame_df)
        mf_df.loc[mf_frame_df.index, ['sf_x_center', 'sf_y_center', 'sf_width', 'sf_height']] = mf_frame_df[
            ['sf_x_center', 'sf_y_center', 'sf_width', 'sf_height']].to_numpy()
    return mf_df


def has_more_than_3_decimal_places(number):
    # Convert the number to a string
    number_str = str(number)

    # Split the string by the decimal point
    parts = number_str.split('.')

    # Check if there are 2 parts and the second part has more than 3 decimals
    return len(parts[1]) >= 3


def input_test(mf_df, sf_df):
    valid_mf = all(has_more_than_3_decimal_places(num) for num in mf_df['score'].to_numpy())
    valid_sf = all(has_more_than_3_decimal_places(num) for num in sf_df['score'].to_numpy())
    return valid_mf and valid_sf


def change_mf_boxes_to_sf_boxes(mf_df, sf_df):
    """
    The following function changes MF boxes to its coupled SF box if it is occluded
    """
    valid_inputs = input_test(mf_df, sf_df)
    if not valid_inputs:
        logger.info(f'Occluded MF boxes are not swithced with coupled SF box')
        return mf_df
    logger.info(f'Switching occluded MF boxes with its coupled SF box')
    mf_df = match_sf_to_mf_boxes(mf_df, sf_df)
    occluded_mf_df = mf_df[(mf_df['is_occluded'] == 1)
                           & (~mf_df['sf_x_center'].isna())]
    mf_df.loc[occluded_mf_df.index, ['x_center', 'y_center', 'width', 'height']] = occluded_mf_df[
        ['sf_x_center', 'sf_y_center', 'sf_width', 'sf_height']].to_numpy()
    mf_df = mf_df.drop(['sf_x_center', 'sf_y_center', 'sf_width', 'sf_height'], axis=1)
    return mf_df


if __name__ == "__main__":
    mf_path = '/home/omer/mnt/server1/workspace/Omer/truncation/65013e677edfb609dc1b3e31/prod_det_mf.tsv'
    sf_path = '/home/omer/mnt/server1/workspace/Omer/truncation/65013e677edfb609dc1b3e31/prod_det_sf.tsv'
    change_mf_boxes_to_sf_boxes(mf_path, sf_path)
