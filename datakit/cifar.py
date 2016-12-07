import numpy as np

from .helpers import data_path
from .helpers import as_iterator

from .constants import ALL

def load(which=ALL, coarse_label=False):
    assert which in (ALL, 'train', 'test')
    if which == ALL: parts = ['train', 'test']
    else: parts = [which]
    data = {}
    for part_name in parts:
        data_part = np.load(data_path('cifar100', part_name))
        X = data_part['data']
        X = X.reshape((X.shape[0], 3, 32, 32))
        if coarse_label is True: y = data_part['coarse_labels']
        else: y = data_part['fine_labels']
        data[part_name] = {'X': X, 'y': y}
    return data

load_as_iterator = lambda which:as_iterator(load(which=which)[which])
