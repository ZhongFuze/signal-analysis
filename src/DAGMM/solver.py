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

        self.sess_run = tf.Session()

    def train(self):
        print("======================TRAIN MODE======================")
        if self.pretrained_model:
            start = int(self.pretrained_model.split('_')[0])
        else:
            start = 1

        # with tf.Session() as sess:
        init = tf.global_variables_initializer()
        self.sess_run.run(init)
        for e in range(start, self.num_epochs):
            input_data = np.random.randn(1024, 118)
            if e % 2 == 0:
                self.sess_run.run(self.dagmm.train_op, feed_dict={self.dagmm.X: input_data})
            else:
                loss = self.sess_run.run(self.dagmm.loss, feed_dict={self.dagmm.X: input_data})
                print loss

        saver = tf.train.Saver()
        saver.save(self.sess_run, self.model_save_path + '/model.ckpt')

    def test(self):
        print("======================TEST MODE======================")
        N = 1024

        saver = tf.train.Saver()
        saver.restore(self.sess_run, self.model_save_path + '/model.ckpt')

        train_input_data = np.random.randn(1024, 118)

        self.sess_run.run(self.dagmm.z, feed_dict={self.dagmm.X: train_input_data})
        self.sess_run.run(self.dagmm.gamma, feed_dict={self.dagmm.X: train_input_data})
        compute_params = self.dagmm.compute_gmm_params(self.dagmm.z, self.dagmm.gamma)
        self.sess_run.run(compute_params, feed_dict={self.dagmm.X: train_input_data})

        gamma_sum = tf.reduce_sum(self.dagmm.gamma, axis=0)
        mu_sum = tf.multiply(self.dagmm.mu, tf.expand_dims(gamma_sum, -1))
        cov_sum = tf.multiply(self.dagmm.cov, tf.expand_dims(tf.expand_dims(gamma_sum, -1), -1))

        train_phi = tf.div(gamma_sum, N)
        train_mu = tf.div(mu_sum, tf.expand_dims(gamma_sum, -1))
        train_cov = tf.div(cov_sum, tf.expand_dims(tf.expand_dims(gamma_sum, -1), -1))

        train_compute_energy = self.dagmm.compute_energy(self.dagmm.z, phi=train_phi, mu=train_mu, cov=train_cov,
                                                         size_average=False)

        train_energy, train_cov_diag = self.sess_run.run(train_compute_energy, feed_dict={self.dagmm.X: train_input_data})
        # train_label = ...

        test_input_data = np.random.randn(100, 118)
        test_compute_energy = self.dagmm.compute_energy(self.dagmm.z, phi=train_phi, mu=train_mu, cov=train_cov,
                                                        size_average=False)
        test_energy, test_cov_diag = self.sess_run.run(test_compute_energy, feed_dict={self.dagmm.X: test_input_data})
        # test_label = ...
        test_label = np.array([1] * 100)

        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        # combined_label = [train_label, test_label]
        thresh = np.percentile(combined_energy, 100 - 20)

        pred = (test_energy > thresh).astype(int)
        gt = test_label.astype(int)

        print 'pred', pred
        print 'real', gt

        from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = prf(gt, pred, average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}".format(accuracy, precision,
                                                                                                    recall, f_score))



