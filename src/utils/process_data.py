#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

import sys
import time
import json
import pickle
import random
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from multiprocessing import Pool


def z_score_normalization(input_x):
    x = np.array(input_x)
    mu = np.average(x)
    sigma = np.std(x)
    x = (x - mu) / sigma
    output_x = list(x)
    return output_x


def min_max_normalization(input_x):
    input_x = map(lambda l: float(l), input_x)
    x = np.array(input_x)
    x_min = np.min(x)
    x_max = np.max(x)
    if (x_max - x_min) == 0.0:
        return input_x
    x = (x - x_min) / (x_max - x_min)
    output_x = list(x)
    return output_x


if __name__ == '__main__':
    in_x = np.random.randint(5, 10, 100)
    in_x = map(lambda x: float(x), in_x)
    out_x = min_max_normalization(in_x)
    print in_x
    print out_x
