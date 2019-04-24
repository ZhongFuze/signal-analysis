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
import banpei
import pandas as pd
from pandas import Series

program_path = '/Users/fuzezhong/Documents/signal-analysis'
data_pkl_path = '{}/{}/{}'.format(program_path, 'data', 'id_65988_key_13_days_30.pkl')
idkey_pkl_path = '{}/{}/{}'.format(program_path, 'data', 'granularity_threshold_math_2.pkl')
sys.path.append(program_path)
from src.utils.process_data import min_max_normalization

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


if __name__ == '__main__':
    event_id = 26080044
    Dtrain = read_all_time_series(idkey_pkl_path, event_id)
    (dims, samples) = np.shape(Dtrain)
    print(Dtrain[:, 0])
    y = Dtrain[:, 0]
    model = banpei.SST(w=120)
    data = Series(y)

    result = model.detect(data, is_lanczos=True)
    plot_x = np.linspace(0, dims, dims)
    plt.plot(plot_x, data)
    plt.plot(plot_x, result)
    plt.show()
