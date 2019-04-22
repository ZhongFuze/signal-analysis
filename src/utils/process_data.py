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
