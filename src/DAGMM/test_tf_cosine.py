#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


def cosine_sim(x1, x2, name='Cosine_Similarity'):
    with tf.name_scope(name):
        x1_val = tf.sqrt(tf.reduce_sum(tf.pow(x1, 2), axis=1))
        x2_val = tf.sqrt(tf.reduce_sum(tf.pow(x2, 2), axis=1))
        denom = tf.multiply(x1_val, x2_val)
        num = tf.reduce_sum(tf.multiply(x1, x2), axis=1)
        print tf.shape(x1)
        a = tf.shape(num)
        return a, tf.div(num, denom + 1e-8)


def euclidean_dis(x1, x2, name='Euclidean_Distance'):
    with tf.name_scope(name):
        diff = tf.subtract(x1, x2)
        euclidean = tf.sqrt(tf.reduce_sum(tf.pow(diff, 2), axis=1))
        return euclidean


X = tf.placeholder(tf.float32, [None, None])
Y = tf.placeholder(tf.float32, [None, None])

sim = cosine_sim(X, Y)
dis = euclidean_dis(X, Y)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    X_test = np.array([[1, 1, 1], [4, 4, 1]])
    Y_test = np.array([[2, 2, 2], [4, 5, 1]])
    sk_sim = cosine_similarity(X_test, Y_test)
    sk_dis = euclidean_distances(X_test, Y_test)

    aa, tf_sim = sess.run(sim, feed_dict={X: X_test, Y: Y_test})
    tf_eud = sess.run(dis, feed_dict={X: X_test, Y: Y_test})
    print '==cosine=='
    print sk_sim
    print tf_sim
    print aa
    print '==euclidean=='
    print sk_dis
    print tf_eud


