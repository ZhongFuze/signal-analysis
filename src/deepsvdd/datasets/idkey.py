#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
"""
Author:         fuzezhong
Filename:       idkey.py
Last modified:
Description:
    idkey数据集
"""
import pickle
import json


class IDKEY_Dataset(object):
    def __init__(self, root, normal_class=0):
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, self.n_classes))
        self.outlier_classes.remove(normal_class)

        self.train_set = None
        self.test_set = None

        # pkl: dict, keys = day_index, value = list(feature)
        self.train_path = '{}/{}/{}'.format(root, 'idkey', 'train_feature.json')
        # txt:
        self.train_label_path = '{}/{}/{}'.format(root, 'idkey', 'train_label.json')

        self.test_path = '{}/{}/{}'.format(root, 'idkey', 'test_feature.json')
        self.test_label_path = '{}/{}/{}'.format(root, 'idkey', 'test_label.json')
        self.train_set = self.load_train()
        self.test_set = self.load_test()

    def load_train(self):
        with open(self.train_path, 'r') as fr:
            train_data = json.load(fr)
        with open(self.train_label_path, 'r') as fr_label:
            train_label = json.load(fr_label)

        data = []
        data_inx = train_data.keys()
        for idx in data_inx:
            r = (train_data[idx], train_label[idx], idx)
            data.append(r)

        return data

    def load_test(self):
        with open(self.test_path, 'r') as fr:
            test_data = json.load(fr)
        with open(self.test_label_path, 'r') as fr_label:
            test_label = json.load(fr_label)

        data = []
        data_inx = test_data.keys()
        for idx in data_inx:
            r = (test_data[idx], test_label[idx], idx)
            data.append(r)

        return data

