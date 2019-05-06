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
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn import (manifold,datasets,decomposition,ensemble,random_projection)
from scipy import interpolate
plt.style.use('ggplot')
font = {'family': 'SimHei'}


program_path = '/Users/fuzezhong/Documents/signal-analysis'
data_pkl_path = '{}/{}/{}'.format(program_path, 'data', 'id_65988_key_13_days_30.pkl')
idkey_pkl_path = '{}/{}/{}'.format(program_path, 'data', 'granularity_threshold_math_2.pkl')
sys.path.append(program_path)

from src.utils.process_data import z_score_normalization, min_max_normalization

def plot_embedding_2d(X,y,title=None):

    # 坐标缩放到[0，1)区间
    x_min,x_max = np.min(X,axis=0),np.max(X,axis=0)
    X = (X - x_min)/(x_max - x_min)
    # 降维后坐标为（X[i，0]，X[i，1]），在该位置画出对应的digits
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for i in range(X.shape[0]):
        # print y[i], type(y[i])
        ax.text(X[i,0],X[i,1],str(i),
                color = plt.cm.Set1((y[i]+1)/10.),
                fontdict={'weight':'bold','size':9})
    if title is not None:
        plt.title(title)


def plot_embedding_3d(X,y,title=None):
    # 坐标缩放到[0，1)区间
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)
    # 降维后坐标为（X[i，0]，X[i，1]，X[i，2]），在该位置画出对应的digits
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1,projection='3d')
    for i in range(X.shape[0]):
        ax.text(X[i, 0], X[i, 1],X[i,2], str(i),
                color=plt.cm.Set1((y[i]+1) / 10.),
                fontdict={'weight': 'bold', 'size': 9})
    if title is not None:
        plt.title(title)


def plot_circle(X, cluster, center):
    (dims, samples) = np.shape(X)
    X_new = []
    for i in range(samples):
        X_new.append(X[:, i])
    X_new.append(center)
    X_new = np.array(X_new)
    y = list(cluster)
    y = map(lambda x: int(x), y)
    y.append(int(2))


    print("Computing PCA projection")
    t0 = time.time()
    X_pca = decomposition.TruncatedSVD(n_components=3).fit_transform(X_new)
    # plot_embedding_2d(X_pca[:, 0:2], y, "PCA 2D")
    plot_embedding_3d(X_pca, y, "PCA 3D (time %.2fs)" % (time.time() - t0))
    plt.show()


def plot_detect(X, cluster, center):
    (dims, samples) = X.shape
    plot_x = np.linspace(0, dims, dims)
    plt.figure(figsize=(20, 6))
    normaly = []
    anomaly = []
    k = 0
    mean_normaly_y = np.zeros(np.shape(X[:, 0]))
    mean_anomaly_y = np.zeros(np.shape(X[:, 0]))
    for i in range(samples):
        k = i
        if cluster[i] == 1.0:
            normaly.append(i)
            plot_y = X[:, i]
            mean_normaly_y += plot_y
            plt.subplot(4, 8, i+1)
            # plt.plot(plot_x, center, color='green')
            plt.plot(plot_x, plot_y, color='blue')

            plt.title('diff_day: ' + str(i))
        if cluster[i] == 0.0:
            plt.subplot(4, 8, i+1)
            anomaly.append(i)
            plot_y = X[:, i]
            mean_anomaly_y += plot_y
            # plt.plot(plot_x, center, color='green')
            plt.plot(plot_x, plot_y, color='red')
            plt.title('diff_day: ' + str(i))

    # mean_normaly_y = min_max_normalization(mean_normaly_y)
    # mean_anomaly_y = min_max_normalization(mean_anomaly_y)
    # plt.subplot(4, 8, k + 2)
    # plt.plot(plot_x, mean_anomaly_y, color='red')
    # plt.plot(plot_x, mean_normaly_y, color='blue')
    # plt.plot(plot_x, center, color='green')
    # plt.title('center and mean')
    # plt.ylim(0.0, 1.0)

    print 'normaly days {0}'.format(normaly)
    print 'anomaly days {0}'.format(anomaly)
    plt.show()


def plot_dynamic_detect(X, cluster, res, center):
    (dims, samples) = X.shape
    plot_x = np.linspace(0, dims, dims)
    plot_x_new = np.linspace(0, dims, 5*dims)
    cnter_func = interpolate.interp1d(plot_x, center, kind='cubic')
    cubic_center = cnter_func(plot_x_new)
    plt.figure(figsize=(8, 10))
    normaly = []
    anomaly = []
    k = 0
    mean_normaly_y = np.zeros(np.shape(X[:, 0]))
    mean_anomaly_y = np.zeros(np.shape(X[:, 0]))
    for i in range(samples):
        k = i
        if cluster[i] == 1.0:
            normaly.append(i)
            func = interpolate.interp1d(plot_x, X[:, i], kind='cubic')
            cubic_y = func(plot_x_new)
            plot_y = X[:, i]
            mean_normaly_y += plot_y
            plt.subplot(8, 3, i+1)
            plt.plot(plot_x_new, cubic_center, color='green')
            plt.plot(plot_x_new, cubic_y, color='blue')
            plt.title('diff_day: %d    res: %.4f' % (7 - i, res[i]), fontsize=10)
            plt.ylim(-1.0, 1.1)
        if cluster[i] == 0.0:
            plt.subplot(8, 3, i+1)
            anomaly.append(i)
            func = interpolate.interp1d(plot_x, X[:, i], kind='cubic')
            cubic_y = func(plot_x_new)
            plot_y = X[:, i]
            mean_anomaly_y += plot_y
            plt.plot(plot_x_new, cubic_center, color='green')
            plt.plot(plot_x_new, cubic_y, color='red')
            plt.title('diff_day: %d    res: %.4f' % (7 - i, res[i]), fontsize=10)
            plt.ylim(-1.0, 1.1)

    # mean_normaly_y = min_max_normalization(mean_normaly_y)
    # mean_anomaly_y = min_max_normalization(mean_anomaly_y)
    # plt.subplot(4, 8, k + 2)
    # plt.plot(plot_x, mean_anomaly_y, color='red')
    # plt.plot(plot_x, mean_normaly_y, color='blue')
    # plt.plot(plot_x, center, color='green')
    # plt.title('center and mean')
    # plt.ylim(0.0, 1.0)

    print 'normaly days {0}'.format(normaly)
    print 'anomaly days {0}'.format(anomaly)
    plt.show()


