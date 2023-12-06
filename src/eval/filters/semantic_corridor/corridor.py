import os
import pandas as pd

from src.eval.filters.semantic_corridor.distance_estimation.distance_estimator import ObjectDistanceEstimator


def filter_non_relevant_objects(df, output_dir, file):
    print('Lateral distance filtering of detection file')
    obj_dist= ObjectDistanceEstimator(df, output_dir, file)
    obj_dist.run()
    df = obj_dist.final_results
    return df