#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

import sys
import time
import json
import pickle
import random
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from multiprocessing import Pool
import luminol
from luminol.anomaly_detector import AnomalyDetector
from luminol.correlator import Correlator

program_path = '/Users/fuzezhong/Documents/signal-analysis'
data_pkl_path = '{}/{}/{}'.format(program_path, 'data', 'id_65988_key_13_days_30.pkl')
idkey_pkl_path = '{}/{}/{}'.format(program_path, 'data', 'granularity_threshold_math_2.pkl')
sys.path.append(program_path)

from src.utils.process_data import z_score_normalization, min_max_normalization
from svdd_sgd import SvddSGD
from src.utils.plot_func import plot_detect


def read_all_time_series(datafile, target_id):
    with open(datafile, 'rb') as fr:
        data = pickle.load(fr)

    target_event_id = target_id
    event_id_list = data.keys()
    points_list = []

    for event_id in event_id_list:
        if event_id == target_event_id:
            points_dict = data[event_id]['points']
            diff_days = points_dict.keys()
            for d in diff_days:
                # points_list.append(min_max_normalization(points_dict[d]['points'][1439-120:1439]))
                points_list.append(points_dict[d]['points'])
            break

    data_X = np.array(points_list)
    X = data_X.T
    print(np.shape(X))
    return X


def transfer_ts(x):
    in_x = list(x)
    ts_dict = {}
    len_ts = len(in_x)
    for i in range(len_ts):
        ts_dict[i] = in_x[i]

    return ts_dict


if __name__ == '__main__':
    event_id = 26258390
    Dtrain = read_all_time_series(idkey_pkl_path, event_id)
    (dims, samples) = np.shape(Dtrain)
    print(Dtrain[:, 0])
    y = Dtrain[:, 0]
    ts = transfer_ts(y)
    # print ts

    my_detector = AnomalyDetector(ts, score_threshold=3.0)
    score = my_detector.get_all_scores()
    score_dict = {}
    for timestamp, value in score.iteritems():
        score_dict[timestamp] = value
    anomalies = my_detector.get_anomalies()

    plot_x = np.linspace(0, dims, dims)
    plt.plot(plot_x, ts.values())
    for a in anomalies:
        print a
        s = a.start_timestamp
        e = a.end_timestamp
        for index in range(s, e + 1):
            plt.plot(index, ts[index], 'rs')

    plt.show()
    # plt.plot(plot_x, score_dict.values())

