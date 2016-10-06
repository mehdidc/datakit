from itertools import imap, cycle

import numpy as np
import os

from helpers import data_path, as_iterator
from constants import *

def load(which=ALL):
    assert which in (ALL, TRAIN, TEST)
    data = {}
    if which in (ALL, TRAIN):
        X_train, y_train = np.load(data_path('mnist', 'train.npy'))
        X_train = list(X_train)
        y_train = list(y_train)
        X_train = np.array(X_train)
        X_train = X_train[:, np.newaxis, :, :]
        y_train = np.array(y_train)
        data[TRAIN] = {'X': X_train, 'y': y_train}
    if which in (ALL, TEST):
        X_test, y_test  = np.load(data_path('mnist', 'test.npy'))
        X_test = list(X_test)
        y_test = list(y_test)
        X_test = np.array(X_test)
        X_test = X_test[:, np.newaxis, :, :]
        y_test = np.array(y_test)
        data[TEST] = {'X': X_test, 'y': y_test}
    return data

load_as_iterator = lambda which:as_iterator(load(which=which)[which])
