#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
"""
Author:         fuzezhong
Filename:       deepSVDD.py
Last modified:
Description:
    deepSVDD类
"""
import sys
import time
import json
import numpy as np
import tensorflow as tf
sys.path.append("/Users/fuzezhong/Documents/signal-analysis")

from src.deepsvdd.network.main import build_network


class DeepSVDD(object):
    def __init__(self, objective='one-class', nu=0.1):
        assert objective in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective

        assert (0 < nu) & (nu <= 1), "For hyperparameter nu, it must hold: 0 < nu <= 1."

        self.nu = nu
        self.R = 0.0  # hypersphere radius R
        self.c = None  # hypersphere center c

        self.model_name = None
        self.model = None

        self.trainer = None
        self.optimizer_name = None

        self.ae_net = None  # autoencoder network for pretraining
        self.ae_trainer = None
        self.ae_optimizer_name = None

        self.results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None,
            'test_scores': None,
        }

    def initial_center(self, train_dataset):
        data, label, idx = train_dataset
        np_data = np.array(data)
        c = np.mean(np_data, axis=0)
        return c

    def set_network(self,
                    model_name,
                    dataset,
                    optimizer_name='adam',
                    lr=0.001,
                    n_epochs=50,
                    lr_milestones=(),
                    batch_size=128,
                    weight_decay=1e-6,
                    device='cpu'):
        """Builds the neural network"""
        self.model_name = model_name
        if self.c is None:
            c = self.initial_center(train_dataset=dataset)
            self.c = c

        self.model = build_network(self.model_name,
                                   self.objective,
                                   self.R,
                                   self.c,
                                   self.nu,
                                   optimizer_name,
                                   lr,
                                   n_epochs,
                                   lr_milestones,
                                   batch_size,
                                   weight_decay,
                                   device)

    def train(self, dataset, optimizer_name='adam', lr=0.001, n_epochs=50, lr_milestones=(), batch_size=128,
              weight_decay=1e-6, device='cpu'):
        self.optimizer_name = optimizer_name



