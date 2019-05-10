#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from model import DaGMM


class Solver(object):
    DEFAULTS = {}

    def __init__(self, data_loader, config):
        # Data loader
        self.__dict__.update(Solver.DEFAULTS, **config)
        self.data_loader = data_loader

        self.dagmm = DaGMM(config, 4, 3)

        # Start with trained model
        if self.pretrained_model:
            # self.load_pretrained_model()
            pass

    def train(self):
        if self.pretrained_model:
            start = int(self.pretrained_model.split('_')[0])
        else:
            start = 1

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            for e in range(start, self.num_epochs):
                input_data = np.random.randn(1024, 118)
                if e % 10 == 0:
                    sess.run(self.dagmm.train_op, feed_dict={self.dagmm.X: input_data})
                else:
                    loss = sess.run(self.dagmm.loss, feed_dict={self.dagmm.X: input_data})
                    print loss