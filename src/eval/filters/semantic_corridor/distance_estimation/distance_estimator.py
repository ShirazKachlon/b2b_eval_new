import os
import numpy as np

from src.eval.filters.semantic_corridor.distance_estimation.distance_estimator_utils import longitudinal_distnace, \
    x_to_angle

LATERAL_DISTANCE_TH = [-5.25, 5.25]
CLASSES = [0, 2001, 2003, 2004]
DISTANCE_TH = {'in_lane': {'lateral_distance': [-2, 2], 'longitudinal_distance': 120},
               'next_lane': {'lateral_distance': [-5.25, 5.25], 'longitudinal_distance': 70}}


class ObjectDistanceEstimator:
    def __init__(self, det_df, output_dir, file):
        self.det_df = det_df
        self.output_dir = output_dir
        self.file = file
        self.init()

    def init(self):
        self.final_results = self.det_df.copy()
        # detection tsv has longitudinal and lateral distances columns
        if self.file == 'det':
            if 'long_dist' not in self.final_results.columns and 'lat_dist' not in self.final_results.columns:
                self.final_results['long_dist'] = np.NAN
                self.final_results['lat_dist'] = np.NAN
                self.flag = 'SF'
            else:
                self.flag = 'MF'
        elif self.file == 'gt':
            if 'long_dist' not in self.final_results.columns and 'lat_dist' not in self.final_results.columns:
                self.final_results['long_dist'] = np.NAN
                self.final_results['lat_dist'] = np.NAN
            self.flag = 'GT'
    
    def run(self):
        if not self.det_df.empty and (self.flag == 'SF' or self.flag == 'GT'):
            print('Calculating objects distances for SF input')
            for label, group_df in self.det_df.groupby('label'):
                if label in CLASSES:
                    indices = group_df.index
                    self._get_distance(indices, label)
        self.filter_by_lateral_dist()

    def _get_distance(self, indices, label):
        longi_dist = longitudinal_distnace(self.det_df.loc[indices, 'height'].to_numpy(), label)
        angles = x_to_angle(self.det_df.loc[indices, 'x_center'].to_numpy())
        lateral_dist = longi_dist * np.tan(angles)
        self.final_results.loc[indices, 'long_dist'] = longi_dist
        self.final_results.loc[indices, 'lat_dist'] = lateral_dist
        path = os.path.join(self.output_dir, 'detections_with_obj_dist.tsv')
        self.final_results.to_csv(path, sep='\t', index=False)

    def filter_by_lateral_dist(self):
        print(f'Filtering object according to lateral distances {LATERAL_DISTANCE_TH} meters')
        indices = self.final_results.index[((self.final_results['lat_dist'] < LATERAL_DISTANCE_TH[0])|
                                            (self.final_results['lat_dist'] > LATERAL_DISTANCE_TH[1]))& 
                                            (self.final_results['label'].isin(CLASSES))]
        
        if self.flag == 'SF' or self.flag == 'MF':
            # removing detection from detection
            self.final_results = self.final_results.drop(indices)
            self.final_results = self.final_results.drop(['long_dist', 'lat_dist'], axis=1)
        else:
            for label in [0,2]:
                if label == 0:
                    ignore_label = 1000
                elif label == 2:
                    ignore_label = 1002
                self.final_results.loc[indices,'label'] = ignore_label
                # fitering gt according to score th
                indices2 = self.final_results.index[(self.final_results['label']==label)&(self.final_results['score'] < 85)]
                self.final_results.loc[indices2,'label'] = ignore_label
            
        
