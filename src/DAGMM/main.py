#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
import argparse
import os
from utils import mkdir
from data_loader import get_data_loader

def str2bool(v):
    return v.lower() in ['true']


def main(config):
    # Create directories if not exist
    mkdir(config.log_path)
    mkdir(config.model_save_path)

    dataset = get_data_loader(data_path=config.data_path, batch_size=config.batch_size, mode=config.mode)

    solver = Solver(dataset, vars(config))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--lr', type=float, default=1e-4)

    # Training settings
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--gmm_k', type=int, default=4)
    parser.add_argument('--latent_dim', type=int, default=3)

    parser.add_argument('--lambda_energy', type=float, default=0.1)
    parser.add_argument('--lambda_cov_diag', type=float, default=0.005)
    parser.add_argument('--pretrained_model', type=str, default=None)

    # Misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])

    # Path
    parser.add_argument('--data_path', type=str, default='kdd_cup.npz')
    parser.add_argument('--log_path', type=str, default='./DAGMM/logs')
    parser.add_argument('--model_save_path', type=str, default='./DAGMM/models')

    # Step size
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=194)
    parser.add_argument('--model_save_step', type=int, default=194)

    config = parser.parse_args()

    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

    main(config)