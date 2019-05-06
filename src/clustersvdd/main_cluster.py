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
data_pkl_path = '{}/{}/{}'.format(program_path, 'data', 'id_65988_key_13_days_30.pkl')
idkey_pkl_path = '{}/{}/{}'.format(program_path, 'data', 'granularity_threshold_math_2.pkl')
convergence_pkl_path = '{}/{}/{}'.format(program_path, 'data', 'dynamic_granularity.pkl')
sys.path.append(program_path)

from src.utils.process_data import z_score_normalization, min_max_normalization
from svdd_sgd import SvddSGD
from src.utils.plot_func import plot_detect, plot_dynamic_detect


def read_time_series(datafile):
    with open(datafile, 'rb') as fr:
        data = pickle.load(fr)

    days = data.keys()
    points_list = []
    for day in days:
        points = data[day]['points']
        points_list.append(min_max_normalization(points))
        # points_list.append(points)

    data_X = np.array(points_list)
    data_X = data_X.T
    # print np.shape(data_X)
    return data_X


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
                points_list.append(min_max_normalization(points_dict[d]['points'][1440-60:1440]))
                # points_list.append(min_max_normalization(points_dict[d]['points']))
            break

    data_X = np.array(points_list)
    X = data_X.T
    print np.shape(X)
    return X


def read_dynamic_time_series(datafile, target_id):
    with open(datafile, 'rb') as fr:
        data = pickle.load(fr)
    event_id_list = data.keys()
    points_list = []
    for event_id in event_id_list:
        if event_id == target_id:
            points_dict = data[event_id]
            now_data = points_dict[5]
            for d in now_data:
                points_list.append(min_max_normalization(d))
            break
    data_X = np.array(points_list)
    X = data_X.T
    print np.shape(X)
    return X


def fit(solver, X, min_chg=0.0, max_iter=40, max_svdd_iter=2000, init_membership=None):
    (dims, samples) = X.shape
    # cinds_old = np.random.randint(0, 2, samples)
    # cinds = np.random.randint(0, 2, samples)
    cinds_old = np.zeros(samples)
    cinds = np.ones(samples)
    if init_membership is not None:
        print('Using init cluster membership.')
        cinds = init_membership

    solver.fits(X)
    print 'svdd initial, time is {0}' \
        .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))

    iter_cnt = 0
    scores = np.zeros(samples)
    # np.sum(np.abs(cinds_old - cinds)) / np.float(samples) > min_chg and
    while np.sum(np.abs(cinds_old - cinds)) / np.float(samples) > min_chg and iter_cnt < max_iter:
        print('Iter={0}'.format(iter_cnt))
        # majorization step
        scores = solver.predicts(X)
        cinds_old = cinds.copy()

        for j in range(samples):
            if abs(scores[j]) < 0.01:
                cinds[j] = 1
            else:
                if scores[j] <= 0.0:
                    cinds[j] = 1
                else:
                    cinds[j] = 0

        print cinds
        solver.fits(X, max_iter=max_svdd_iter)
        iter_cnt += 1
        print 'svdd training finished after {0} iterations, time is {1}'\
            .format(iter_cnt, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))

    return cinds


def predict_now(solver, X):
    scores = solver.predicts(X)
    cinds = 0.0

    if abs(scores) < 0.01:
        cinds = 1.0
    else:
        if scores <= 0.0:
            cinds = 1.0
        else:
            cinds = 0.0

    return scores, cinds


def predict(solver, X):
    (dims, samples) = X.shape
    scores = solver.predicts(X)
    cinds = np.zeros(samples)

    # for j in range(samples):
    #     if scores[j] <= 0.0:
    #         cinds[j] = 1
    #     else:
    #         cinds[j] = 0

    for j in range(samples):
        if abs(scores[j]) < 0.01:
            cinds[j] = 1
        else:
            if scores[j] <= 0.0:
                cinds[j] = 1
            else:
                cinds[j] = 0

    # negative = []
    # positive = []
    # for i in range(samples):
    #     if scores[i] <= 0.0:
    #         negative.append(scores[i])
    #     else:
    #         positive.append(scores[i])
    #
    # if len(negative) > len(positive):
    #     negative.sort()
    #     index = int(len(negative) * 0.8)
    #     threshold = (negative[index])
    # else:
    #     threshold = -0.01
    #
    # print 'threshold', threshold
    # for j in range(samples):
    #     if scores[j] < threshold:
    #         cinds[j] = 1
    #     else:
    #         cinds[j] = 0
    #
    return scores, cinds


def train(data, nu, membership):
    svdd_solver = SvddSGD(data, nu)
    cinds = fit(svdd_solver, data, nu, init_membership=membership)
    return svdd_solver, cinds


if __name__ == '__main__':
    event_id = 26104924
    cluster = 1
    nu = 0.1
    membership = None
    # membership = read_label(data_label_path)
    # Dtrain = read_time_series(data_pkl_path)
    # Dtrain = read_all_time_series(idkey_pkl_path, event_id)  # 未聚合
    D = read_dynamic_time_series(convergence_pkl_path, event_id)  # 已聚合
    (dims, samples) = np.shape(D)
    Dtrain = D[:, 0:samples-1]
    Dtest = D[:, samples-1]
    svdd, cinds = train(Dtrain, nu, membership)
    res, cluser_res = predict(svdd, Dtrain)
    test_res, test_cluser_res = predict_now(svdd, Dtest)

    res_list = list(res)
    res_list.append(test_res)
    cluser_res_list = list(cluser_res)
    cluser_res_list.append(test_cluser_res)

    for r in range(D.shape[1]):
        print r, res_list[r]

    plot_dynamic_detect(D, cluser_res_list, res_list, svdd.c)

    print 'finished'
