from helpers import data_path, dict_apply, dict_apply_recur
import numpy as np
import os

def load(which='all'):
    assert which in ('all', 'train', 'test')

    data = {}
    if which in ('all', 'train'):
        X_train, y_train = np.load(data_path('mnist', 'train.npy'))
        X_train = list(X_train)
        y_train = list(y_train)
        X_train = np.array(X_train)
        X_train = X_train[:, np.newaxis, :, :]
        y_train = np.array(y_train)
        data['train'] = {'X': X_train, 'y': y_train}
    if which in ('all', 'test'):
        X_test, y_test  = np.load(data_path('mnist', 'test.npy'))
        X_test = list(X_test)
        y_test = list(y_test)
        X_test = np.array(X_test)
        X_test = X_test[:, np.newaxis, :, :]
        y_test = np.array(y_test)
        data['test'] = {'X': X_test, 'y': y_test}
    return data
