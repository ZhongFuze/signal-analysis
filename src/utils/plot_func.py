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

program_path = '/Users/fuzezhong/Documents/signal-analysis'
data_path = '{}/{}/{}'.format(program_path, 'data', 'id_24048_key_16_days_30.pkl')

if __name__ == '__main__':
    print data_path
    with open(data_path, 'rb') as fr:
        data = pickle.load(fr)

    days = data.keys()
    data_point_num = len(data[days[0]]['points'])
    data_point_x = np.linspace(0, data_point_num, data_point_num)
    for day in days:
        points = data[day]['points']
        plt.plot(data_point_x, points)

    plt.show()
