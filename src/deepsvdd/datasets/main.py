#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
"""
Author:         fuzezhong
Filename:       main.py
Last modified:
Description:
    load数据集主函数
"""
import sys
sys.path.append("/Users/fuzezhong/Documents/signal-analysis")
from src.deepsvdd.datasets.mnist import MNIST_Dataset
from src.deepsvdd.datasets.idkey import IDKEY_Dataset


def load_dataset(dataset_name, data_path, normal_class):
    """Loads the dataset."""

    implemented_datasets = ('mnist', 'idkey')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'mnist':
        dataset = MNIST_Dataset(root=data_path, normal_class=normal_class)

    if dataset_name == 'idkey':
        dataset = IDKEY_Dataset(root=data_path, normal_class=normal_class)

    return dataset
