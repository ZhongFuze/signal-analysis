#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
"""
Author:         fuzezhong
Filename:       main.py
Last modified:
Description:
    deepSVDD主函数
"""

import sys
import time
import json
sys.path.append("/Users/fuzezhong/Documents/signal-analysis")
import argparse

from src.deepsvdd.datasets.main import load_dataset
from src.deepsvdd.deepSVDD import DeepSVDD

def runner():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='idkey', help='idkey or mnist')
    parser.add_argument('--n_jobs_dataloader', default=0, help='Number of workers for data loading')
    parser.add_argument('--model_name', default='idkey_net', help='idkey_net or mnist_net')
    parser.add_argument('--xp_path', default='/Users/fuzezhong/Documents/signal-analysis/src/deepsvdd/experiments/idkey',
                        help='Experiments path')
    parser.add_argument('--data_path',
                        default='/Users/fuzezhong/Documents/signal-analysis/data',
                        help='Dataset path')
    parser.add_argument('--load_config', default=None, help='Json config file (default: None).')
    parser.add_argument('--load_model', default=None, help='Model file path (default: None).')
    parser.add_argument('--objective', default='one-class', help='one-class or soft-boundary')
    parser.add_argument('--normal_class', default=0, help='Specify the normal class label of the dataset')

    parser.add_argument('--nu', default=0.1, help='Deep SVDD hyperparameter nu (must be 0 < nu <= 1).')
    parser.add_argument('--device', default='cpu', help='cpu,gpu or cuda')
    parser.add_argument('--seed', default=-1, help='Set seed. If -1, use randomization.')
    parser.add_argument('--optimizer_name', default='adam', help='the optimizer to use for Deep SVDD network training.')
    parser.add_argument('--lr', default=0.001, help='learning rate for Deep SVDD network training. Default=0.001')
    parser.add_argument('--n_epochs', default=50, help='Number of epochs to train.')
    parser.add_argument('--lr_milestone', default=0, help='Lr scheduler milestones')
    parser.add_argument('--batch_size', default=128, help='Batch size for mini-batch training.')
    parser.add_argument('--weight_decay', default=1e-6, help='L2 penalty hyperparameter for Deep SVDD objective.')

    parser.add_argument('--pretrain', default=False, help='Pretrain neural network parameters via autoencoder.')
    parser.add_argument('--ae_optimizer_name', default='adam', help='optimizer to use for autoencoder pretraining.')
    parser.add_argument('--ae_lr', default=0.001, help='learning rate for autoencoder pretraining. Default=0.001')
    parser.add_argument('--ae_n_epochs', default=100, help='Number of epochs to train autoencoder.')
    parser.add_argument('--ae_lr_milestone', default=0, help='Autoencoder lr scheduler milestones ')
    parser.add_argument('--ae_batch_size', default=128, help='Batch size for mini-batch autoencoder training.')
    parser.add_argument('--ae_weight_decay', default=1e-6, help='L2 penalty hyperparameter for autoencoder objective.')

    args = parser.parse_args()

    # load data
    dataset = load_dataset(args.dataset, args.data_path, args.normal_class)

    # initial model
    deep_SVDD = DeepSVDD(args.objective, args.nu)
    deep_SVDD.set_network(args.model_name,
                          args.dataset,
                          args.optimizer_name,
                          args.lr,
                          args.n_epochs,
                          args.lr_milestones,
                          args.batch_size,
                          args.weight_decay,
                          args.device)

    # train model
    deep_SVDD.train(dataset,
                    args.optimizer_name,
                    args.lr,
                    args.n_epochs,
                    args.lr_milestone,
                    args.batch_size,
                    args.weight_decay,
                    args.device)
    # test model
    deep_SVDD.test(dataset, args.device)

    # save model
    # ...
