import numpy as  np
import os

DATA_PATH = os.getenv('DATA_PATH', '.')

def data_path(*args):
    args = (DATA_PATH,) + args
    return os.path.join(*args)

class BatchIterator(object):

    def __init__(self, data,
                 shuffle=False, 
                 random_state=None):
        self.data = data
        self.shuffle = shuffle
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def flow(self, batch_size=128, repeat=True):
        k = self.data.keys()[0]
        nb_examples = len(self.data[k])
        indices = np.arange(nb_examples)
        while True:
            if self.shuffle: self.rng.shuffle(indices)
            for i in range(0, nb_examples, batch_size):
                excerpt = indices[i:i + batch_size]
                data_excerpt = dict_map(self.data, lambda values:values[excerpt])
                yield data_excerpt
            if repeat is False:
                break

def dict_apply_recur(d, fn, cols=None):
    return {k: dict_apply(v, fn, cols=cols) for k,v in d.items()}

def dict_apply(d, fn, cols=None):
    if not cols: cols = d.keys()
    d = d.copy()
    d.update({k: fn(d[k]) for k in cols})
    return d
