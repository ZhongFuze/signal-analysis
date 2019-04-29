#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
"""
Author:         fuzezhong
Filename:       main.py
Last modified:
Description:
    mnist构造网络
"""
import sys
sys.path.append("/Users/fuzezhong/Documents/signal-analysis")


class MNIST_Net(object):
    def __init__(self):
        self.param = 0

    def forward(self, x):
        x = self.param
        return x
