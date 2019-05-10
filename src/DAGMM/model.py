#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from utils import cosine_sim, euclidean_dis


class DaGMM(object):
    DEFAULTS = {}

    def __init__(self, config, n_gmm=2, latent_dim=3):
        super(DaGMM, self).__init__()
        # Data loader
        self.__dict__.update(DaGMM.DEFAULTS, **config)
        self.X = tf.placeholder("float", [self.batch_size, 118], name='input')

        self.dims = 118
        self.n_hidden_1 = 60
        self.n_hidden_2 = 30
        self.n_hidden_3 = 10
        self.n_hidden_4 = 1

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
            'estimation_h2': tf.Variable(tf.random_normal([self.n_hidden_3, n_gmm])),

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
            'estimation_b2': tf.Variable(tf.random_normal([n_gmm])),

        }

        self.enc = self.encoder(self.X, self.weights, self.biases)
        self.dec = self.decoder(self.enc, self.weights, self.biases)

        rec_cosine = cosine_sim(self.X, self.dec)
        rec_euclidean = euclidean_dis(self.X, self.dec)
        self.u_rec_cosine = tf.expand_dims(rec_cosine, -1)
        self.u_rec_euclidean = tf.expand_dims(rec_euclidean, -1)
        self.z = tf.concat([self.enc, self.u_rec_cosine, self.u_rec_euclidean], axis=1)
        self.gamma = self.estimation(self.z, self.weights, self.biases)

        e = tf.subtract(self.X, self.dec)
        recon_error = tf.reduce_mean(tf.pow(e, 2))

        self.phi, self.mu, self.cov = self.compute_gmm_params(self.z, self.gamma)

        sample_energy, cov_diag = self.compute_energy(self.z, self.phi, self.mu, self.cov)

        self.loss = recon_error + self.lambda_energy * sample_energy + self.lambda_cov_diag * cov_diag

        self.optimizer = tf.train.AdamOptimizer(1e-4)

        self.train_op = self.optimizer.minimize(self.loss)

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
    def compute_gmm_params(z, gamma):
        # K (n_gmm)
        N = tf.shape(gamma)[0]
        sum_gamma = tf.reduce_sum(gamma, axis=0)
        # phi = tf.reduce_mean(gamma, axis=0)
        phi = tf.div(sum_gamma, tf.cast(N, dtype=tf.float32))

        # continue
        u_gamma = tf.expand_dims(gamma, -1)
        u_z = tf.expand_dims(z, 1)
        sum_mu = tf.reduce_sum(tf.matmul(u_gamma, u_z), axis=0)

        u_sum_gamma = tf.expand_dims(sum_gamma, -1)
        mu = tf.div(sum_mu, u_sum_gamma)

        z_mu = tf.subtract(u_z, tf.expand_dims(mu, 0))
        z_mu_outer = tf.matmul(tf.expand_dims(z_mu, -1), tf.expand_dims(z_mu, -2))

        uu_gamma = tf.expand_dims(tf.expand_dims(gamma, -1), -1)
        uu_sum_gamma = tf.expand_dims(u_sum_gamma, -1)

        # ????
        sum_cov = tf.reduce_sum(tf.multiply(uu_gamma, z_mu_outer), axis=0)

        cov = tf.div(sum_cov, uu_sum_gamma)

        return phi, mu, cov

    @staticmethod
    def compute_energy(z, phi, mu, cov, size_average=True):
        k, D, _ = cov.shape
        eye_D = tf.constant(D, dtype=tf.int32)

        z_mu = tf.subtract(tf.expand_dims(z, 1), tf.expand_dims(mu, 0))

        cov_inverse = []
        det_cov = []
        cov_diag = 0
        eps = 1e-12

        for i in range(k):
            # cov_k = tf.add(cov[i], tf.multiply(eps, tf.eye(eye_D)))
            cov_k = cov[i]
            cov_inverse.append(tf.expand_dims(tf.matrix_inverse(cov_k), 0))
            # cholesky
            tmp = tf.cholesky(tf.multiply(2.0 * np.pi, cov_k))
            det_cov.append(tf.expand_dims(tf.reduce_prod(tf.diag(tmp)), 0))

            cov_diag = cov_diag + tf.reduce_sum(tf.div(1.0, tf.diag(cov_k)))

        cov_inverse = tf.concat(cov_inverse, axis=0)
        det_cov = tf.concat(det_cov, axis=0)

        u_z_mu = tf.expand_dims(z_mu, -1)
        u_cov_inverse = tf.expand_dims(cov_inverse, 0)

        tmp_mul = tf.multiply(u_z_mu, u_cov_inverse)
        reduce_tmp_mul = tf.reduce_sum(tmp_mul, axis=-2)
        tmp_mul_again = tf.multiply(reduce_tmp_mul, z_mu)
        reduce_tmp_mul_again = tf.reduce_sum(tmp_mul_again, axis=-1)
        exp_term_tmp = tf.multiply(-0.5, reduce_tmp_mul_again)

        max_val = tf.reduce_max(tf.nn.relu(exp_term_tmp), axis=1)
        exp_term = tf.exp(exp_term_tmp - tf.expand_dims(max_val, -1))

        u_phi = tf.expand_dims(phi, 0)
        u_det_cov = tf.expand_dims(tf.sqrt(det_cov), 0)

        in_log = tf.add(tf.reduce_sum(tf.div(tf.multiply(u_phi, exp_term), u_det_cov), axis=1), eps)
        s_max_val = tf.squeeze(max_val)
        negative_max_val = tf.subtract(0.0, s_max_val)
        sample_energy = tf.subtract(negative_max_val, tf.log(in_log))

        if size_average:
            sample_energy = tf.reduce_mean(sample_energy)

        return sample_energy, cov_diag

    # def forward(self, X):
    #     # forward
    #     enc = self.encoder(X, self.weights, self.biases)
    #     dec = self.decoder(enc, self.weights, self.biases)
    #
    #     rec_cosine = cosine_sim(X, dec)
    #     rec_euclidean = euclidean_dis(X, dec)
    #     z = tf.concat([enc, rec_cosine, rec_euclidean], axis=1)
    #     gamma = self.estimation(z, self.weights, self.biases)
    #
    #     return enc, dec, z, gamma
    #
    # # enc, dec, z, gamma = self.dagmm(input_data)
    # # total_loss, sample_energy, recon_error, cov_diag =
    # # self.dagmm.loss_function(input_data, dec, z, gamma, self.lambda_energy, self.lambda_cov_diag)
    # def loss_function(self, X, X_hat, z, gamma):
    #     e = tf.subtract(X, X_hat)
    #     recon_error = tf.reduce_mean(tf.pow(e, 2))
    #     self.phi, self.mu, self.cov = self.compute_gmm_params(z, gamma)
    #
    #     sample_energy, cov_diag = self.compute_energy(z, self.phi, self.mu, self.cov)
    #
    #     self.loss = recon_error + self.lambda_energy * sample_energy + self.lambda_cov_diag * cov_diag
    #
    #     return sample_energy, recon_error, cov_diag
