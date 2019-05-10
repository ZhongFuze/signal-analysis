#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
import os
import tensorflow as tf


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def cosine_sim(x1, x2):
    x1_val = tf.sqrt(tf.reduce_sum(tf.pow(x1, 2), axis=1))
    x2_val = tf.sqrt(tf.reduce_sum(tf.pow(x2, 2), axis=1))
    denom = tf.multiply(x1_val, x2_val)
    num = tf.reduce_sum(tf.multiply(x1, x2), axis=1)
    return tf.div(num, denom + 1e-8)


def euclidean_dis(x1, x2):
    diff = tf.subtract(x1, x2)
    euclidean = tf.sqrt(tf.reduce_sum(tf.pow(diff, 2), axis=1))
    return euclidean
