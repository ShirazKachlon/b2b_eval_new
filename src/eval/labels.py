class _2w_label(object):
    def det_classes(self):
        return ['2w', '1']

    def manual_gt_classes(self):
        return ['BIKE', 'MOTOR', 'BIKE_CIPV', 'MOTOR_CIPV', 'MOTOR_NLV', 'BIKE_NLV', 'bike', 'motor', 'bike_cipv',
                'motor_cipv', 'motor_nlv', 'bike_nlv']

    def al_gt_classes(self):
        return ['2w']

    def manual_gt_ignore_classes(self):
        return ['GENERAL_IGNORE', 'DONTCARE', 'IGNORE', 'MISC', 'IGNORE_BICYCLE', 'BIKE_WITHOUT_RIDER',
                'MOTOR_WITHOUT_RIDER', 'TRICYCLE', 'KICK_SCOOTER', 'BABY_CARRIAGE', 'tricycle_nlv', 'tricycle_cipv',
                'kick_scooter_cipv', 'kick_scooter_nlv', 'TRICYCLE_NLV', 'TRICYCLE_CIPV', 'KICK_SCOOTER_CIPV',
                'KICK_SCOOTER_NLV', 'general_ignore', 'dontcare', 'ignore', 'misc', 'ignore_bicycle',
                'bike_without_rider', 'motor_without_rider', 'tricycle', 'kick_scooter', 'baby_carriage']

    def al_gt_ignore_classes(self):
        return ['2w_lanes_ignore']


class _4w_label(object):
    def det_classes(self):
        return ['4w', '2', '2001', '2003', '2004']

    def manual_gt_classes(self):
        return ['BUS', 'BUS_CIPV', 'BUS_NLV', 'CAR', 'CAR_CIPV', 'CAR_NLV', 'PICK_UP_CAR', 'PICK_UP_CAR_CIPV',
                'PICK_UP_CAR_NLV', 'TOW_TRUCK', 'TOW_TRUCK_NLV', 'TOW_TRUCK_CIPV', 'TRUCK', 'TRUCK_CIPV', 'TRUCK_NLV',
                'VAN', 'VAN_CIPV', 'VAN_NLV', 'DUMMY_CAR', 'TRAILER', 'trailer', 'construction_vehicle',
                'construction_vehicle_nlv', 'construction_vehicle_cipv', 'CONSTRUCTION_VEHICLE',
                'CONSTRUCTION_VEHICLE_NLV', 'CONSTRUCTION_VEHICLE_CIPV', 'RV', 'MINI_TRUCK', 'bus', 'bus_cipv',
                'bus_nlv', 'car', 'car_cipv', 'car_nlv', 'pick_up_car', 'pick_up_car_cipv', 'pick_up_car_nlv',
                'tow_truck', 'tow_truck_nlv', 'tow_truck_cipv', 'truck', 'truck_cipv', 'truck_nlv', 'van', 'van_cipv',
                'van_nlv', 'dummy_car', 'rv', 'mini_truck']

    def al_gt_classes(self):
        return ['4w', '4w_no_road_contact', '4w_out_of_distance', '4w_under_score_85', '4w_lanes_ignore']

    def manual_gt_ignore_classes(self):
        return ['DONTCARE', 'GENERAL_IGNORE', 'IGNORE', 'IGNORE_VEHICLE', 'MISC', 'OPEN_CAR_DOOR', 'TRAIN', 'dontcare',
                'general_ignore', 'ignore', 'ignore_vehicle', 'misc', 'open_car_door', 'train']

    def al_gt_ignore_classes(self):
        return ['4w_fa_dbox', '4w_no_road_contact', '4w_out_of_distance', '4w_under_score_85', '4w_lanes_ignore']


class ped_label(object):
    def det_classes(self):
        return ['ped', '0']

    def manual_gt_classes(self):
        return ['PEDESTRIAN', 'pedestrian', 'pedestrian_cipv', 'pedestrian_nlv', 'pedestrian_with_wheelchair',
                'pedestrian_with_walking_aid', 'PEDESTRIAN_CIPV', 'PEDESTRIAN_NLV', 'PEDESTRIAN_WITH_WHEELCHAIR',
                'PEDESTRIAN_WITH_WALKING_AID', 'pedestrian_with_cane', 'pedestrian_with_stroller',
                'pedestrian_with_walker', 'pedestrian_with_wheelchair_cipv', 'pedestrian_with_wheelchair_nlv',
                'pedestrian_with_walking_aid_cipv', 'pedestrian_with_walking_aid_nlv', 'pedestrian_with_cane_cipv',
                'pedestrian_with_cane_nlv', 'pedestrian_with_stroller_cipv', 'pedestrian_with_stroller_nlv',
                'pedestrian_with_walker_cipv', 'pedestrian_with_walker_nlv', 'pedestrian_with_wheelchair_cipv',
                'pedestrian_with_wheelchair_nlv', 'pedestrian_with_walking_aid_cipv', 'pedestrian_with_walking_aid_nlv',
                'pedestrian_with_cane_cipv', 'pedestrian_with_cane_nlv', 'pedestrian_with_stroller_cipv',
                'pedestrian_with_stroller_nlv', 'pedestrian_with_walker_cipv', 'pedestrian_with_walker_nlv',
                'pedestrian_with_wheelchair', 'pedestrian_with_walking_aid', 'pedestrian_with_cane',
                'pedestrian_with_stroller', 'pedestrian_with_walker']

    def al_gt_classes(self):
        return ['ped', 'ped_out_of_distance', 'ped_under_score_85', 'ped_lanes_ignore']

    def manual_gt_ignore_classes(self):
        return ['CHILD', 'CHILD_MCP', 'child', 'child_mcp', 'DONTCARE', 'GENERAL_IGNORE', 'IGNORE', 'IGNORE_PEDESTRIAN',
                'MISC', 'dontcare', 'general_ignore', 'ignore', 'ignore_pedestrian', 'misc', 'RIDER', 'RIDER_NLV',
                'RIDER_CIPV', 'BABY_CARRIAGE']

    def al_gt_ignore_classes(self):
        return ['ped_lanes_ignore', 'ped_out_of_distance', 'ped_under_score_85', 'ped_fa_pole', 'ped_in_window']


labels_if_autolabeling_gt = {
    '2w': {
        'det_classes': _2w_label().det_classes(),
        'gt_classes': _2w_label().al_gt_classes(),
        'gt_ignore_classes': _2w_label().al_gt_ignore_classes(),
    },
    '4w': {
        'det_classes': _4w_label().det_classes(),
        'gt_classes': _4w_label().al_gt_classes(),
        'gt_ignore_classes': _4w_label().al_gt_ignore_classes(),
    },
    'peds': {
        'det_classes': ped_label().det_classes(),
        'gt_classes': ped_label().al_gt_classes(),
        'gt_ignore_classes': ped_label().al_gt_ignore_classes(),
    }
}

labels_if_autolabeling_det = {
    '2w': {
        'det_classes': _2w_label().det_classes(),
        'gt_classes': _2w_label().manual_gt_classes(),
        'gt_ignore_classes': _2w_label().manual_gt_ignore_classes(),
    },
    '4w': {
        'det_classes': _4w_label().det_classes(),
        'gt_classes': _4w_label().manual_gt_classes(),
        'gt_ignore_classes': _4w_label().manual_gt_ignore_classes(),
    },
    'peds': {
        'det_classes': ped_label().det_classes(),
        'gt_classes': ped_label().manual_gt_classes(),
        'gt_ignore_classes': ped_label().manual_gt_ignore_classes(),
    }
}
