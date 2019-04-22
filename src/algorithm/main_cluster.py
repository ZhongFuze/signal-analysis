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
sys.path.append(program_path)

from src.utils.process_data import z_score_normalization
from svdd_sgd import SvddSGD


def read_time_series(datafile):
    with open(data_path, 'rb') as fr:
        data = pickle.load(fr)

    days = data.keys()
    points_list = []
    for day in days:
        points = data[day]['points']
        # points_list.append(z_score_normalization(points))
        points_list.append(points)

    data_X = np.array(points_list)
    data_X = data_X.T
    print np.shape(data_X)
    return data_X


def fit(solver, X, min_chg=0.0, max_iter=40, max_svdd_iter=2000, init_membership=None):
    (dims, samples) = X.shape
    cinds_old = np.random.randint(0, 2, samples)
    cinds = np.random.randint(0, 2, samples)
    if init_membership is not None:
        print('Using init cluster membership.')
        cinds = init_membership

    solver.fits(X)
    print 'svdd initial, time is {0}' \
        .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))

    iter_cnt = 0
    scores = np.zeros(samples)
    while np.sum(np.abs(cinds_old - cinds)) / np.float(samples) > min_chg and iter_cnt < max_iter:
        print('Iter={0}'.format(iter_cnt))
        # majorization step
        scores = solver.predicts(X)
        cinds_old = cinds

        for i in range(samples):
            if scores[i] >= 0.0:
                cinds[i] = 0
            else:
                cinds[i] = 1

        solver.fits(X, max_iter=max_svdd_iter)
        iter_cnt += 1
        print 'svdd training finished after {0} iterations, time is {1}'\
            .format(iter_cnt, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))

    return cinds


def predict(solver, X):
    (dims, samples) = X.shape
    scores = solver.predict(X)
    cinds = np.zeros(samples)
    for i in range(samples):
        if scores[i] >= 0.0:
            cinds[i] = 0
        else:
            cinds[i] = 1

    return scores, cinds


def train(data, nu, membership):
    svdd_solver = SvddSGD(nu)
    cinds = fit(svdd_solver, data, nu, init_membership=membership)
    return svdd_solver, cinds


if __name__ == '__main__':
    cluster = 1
    nu = 0.1
    membership = None
    Dtrain = read_time_series(data_path)
    svdd, cinds = train(Dtrain, nu, membership)
    res, cluser_res = predict(svdd, Dtrain)

    print 'finished'




