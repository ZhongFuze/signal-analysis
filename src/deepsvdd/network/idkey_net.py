#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
"""
Author:         fuzezhong
Filename:       idkey_net.py
Last modified:
Description:
    idkey构造网络
"""
import tensorflow as tf
import sys
sys.path.append("/Users/fuzezhong/Documents/signal-analysis")


class IDKEY_Net(object):
    def __init__(self,
                 objective,
                 R,
                 c,
                 nu,
                 optimizer_name='adam',
                 lr=0.001,
                 n_epochs=150,
                 lr_milestones=(),
                 n_batch_size=128,
                 weight_decay=1e-6,
                 device='cpu'):

        self.learning_rate = tf.placeholder(name='learning_rate', dtype=tf.float32)
        self.input = tf.placeholder(name='input_feature', dtype=tf.float32)
        self.label = tf.placeholder(name='input_label', dtype=tf.float32)
        self.nu_penalty = tf.constant(nu)
        self.radius = tf.Variable(R, name='radius')
        self.center = tf.Variable(c, name='center')







