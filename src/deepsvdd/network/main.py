#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
"""
Author:         fuzezhong
Filename:       main.py
Last modified:
Description:
    构造网络
"""
import sys
sys.path.append("/Users/fuzezhong/Documents/signal-analysis")
from src.deepsvdd.network.idkey_net import IDKEY_Net
from src.deepsvdd.network.mnist_net import MNIST_Net


def build_network(model_name,
                  objective,
                  R,
                  c,
                  nu,
                  optimizer_name,
                  lr,
                  n_epochs,
                  lr_milestones,
                  batch_size,
                  weight_decay,
                  device):
    """Builds the neural network."""

    implemented_networks = ('mnist_LeNet', 'cifar10_LeNet')
    assert model_name in implemented_networks

    model = None

    if model_name == 'mnist_net':
        model = MNIST_Net()

    if model_name == 'idkey_net':
        model = IDKEY_Net(objective,
                          R,
                          c,
                          nu,
                          optimizer_name,
                          lr,
                          n_epochs,
                          lr_milestones,
                          batch_size,
                          weight_decay,
                          device)

    return model
