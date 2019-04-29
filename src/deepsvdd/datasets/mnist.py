#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
"""
Author:         fuzezhong
Filename:       mnist.py
Last modified:
Description:
    mnist数据集
"""


class MNIST_Dataset(object):
    def __init__(self, root, normal_class=0):
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 10))
        self.outlier_classes.remove(normal_class)

        self.train_set = None
        self.test_set = None
