#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from utils import cosine_sim, euclidean_dis


class Solver(object):
    DEFAULTS = {}

    def __init__(self, dataset, config, n_gmm = 2, latent_dim=3):
        # Data loader
        self.__dict__.update(Solver.DEFAULTS, **config)
        self.data_loader = dataset

        # Start with trained model
        if self.pretrained_model:
            # self.load_pretrained_model()
            pass

        self.dims = 60
        self.n_hidden_1 = 60
        self.n_hidden_2 = 30
        self.n_hidden_3 = 10
        self.n_hidden_4 = 1

        X = tf.placeholder("float", [None, self.dims])

        self.phi = tf.Variable(tf.zeros([n_gmm]))
        self.mu = tf.Variable(tf.zeros([n_gmm, latent_dim]))
        self.cov = tf.Variable(tf.zeros([n_gmm, latent_dim, latent_dim]))

        self.weights = {
            'encoder_h1': tf.Variable(tf.random_normal([self.dims, self.n_hidden_1])),
            'encoder_h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2])),
            'encoder_h3': tf.Variable(tf.random_normal([self.n_hidden_2, self.n_hidden_3])),
            'encoder_h4': tf.Variable(tf.random_normal([self.n_hidden_3, self.n_hidden_4])),

            'decoder_h1': tf.Variable(tf.random_normal([self.n_hidden_4, self.n_hidden_3])),
            'decoder_h2': tf.Variable(tf.random_normal([self.n_hidden_3, self.n_hidden_2])),
            'decoder_h3': tf.Variable(tf.random_normal([self.n_hidden_2, self.n_hidden_1])),
            'decoder_h4': tf.Variable(tf.random_normal([self.n_hidden_1, self.dims])),

            'estimation_h1': tf.Variable(tf.random_normal([latent_dim, self.n_hidden_3])),
            'estimation_h2': tf.Variable(tf.random_normal([self.n_hidden_3, self.n_gmm])),

        }

        self.biases = {
            'encoder_b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
            'encoder_b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
            'encoder_b3': tf.Variable(tf.random_normal([self.n_hidden_3])),
            'encoder_b4': tf.Variable(tf.random_normal([self.n_hidden_4])),

            'decoder_b1': tf.Variable(tf.random_normal([self.n_hidden_3])),
            'decoder_b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
            'decoder_b3': tf.Variable(tf.random_normal([self.n_hidden_1])),
            'decoder_b4': tf.Variable(tf.random_normal([self.dims])),

            'estimation_b1': tf.Variable(tf.random_normal([self.n_hidden_3])),
            'estimation_b2': tf.Variable(tf.random_normal([self.n_gmm])),

        }

        # forward
        self.enc = self.encoder(X, self.weights, self.biases)
        self.dec = self.decoder(self.enc, self.weights, self.biases)

        # rec_cosine = cosine_similarity(X, dec)
        self.rec_cosine = cosine_sim(X, self.dec)
        self.rec_euclidean = euclidean_dis(X, self.dec)

        self.z = tf.concat([self.enc, self.rec_cosine, self.rec_euclidean], axis=1)

        self.gamma = self.estimation(self.z, self.weights, self.biases)


    @staticmethod
    def encoder(x, weights, biases):
        layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
        layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
        layer_3 = tf.nn.tanh(tf.add(tf.matmul(layer_2, weights['encoder_h3']), biases['encoder_b3']))
        layer_4 = tf.nn.tanh(tf.add(tf.matmul(layer_3, weights['encoder_h4']), biases['encoder_b4']))
        return layer_4

    @staticmethod
    def decoder(x, weights, biases):
        layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
        layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
        layer_3 = tf.nn.tanh(tf.add(tf.matmul(layer_2, weights['decoder_h3']), biases['decoder_b3']))
        layer_4 = tf.nn.tanh(tf.add(tf.matmul(layer_3, weights['decoder_h4']), biases['decoder_b4']))
        return layer_4

    @staticmethod
    def estimation(x, weights, biases):
        layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['estimation_h1']), biases['estimation_b1']))
        layer_1 = tf.layers.dropout(layer_1, rate=0.5)
        layer_2 = tf.nn.softmax(tf.add(tf.matmul(layer_1, weights['estimation_h2']), biases['estimation_b2']))
        return layer_2

    @staticmethod
    def loss_function(x, x_hat, z, gamma, lambda_energy, lambda_cov_diag):
        e = tf.subtract(x, x_hat)
        recon_error = tf.reduce_mean(tf.pow(e, 2))
        phi, mu, cov = compute_gmm_params(z, gamma)

    def compute_gmm_params(self):
        # K (n_gmm)
        self.phi = tf.reduce_mean(self.gamma, axis=0)
        # continue

