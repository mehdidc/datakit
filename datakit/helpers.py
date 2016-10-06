from itertools import imap, chain, repeat

import numpy as  np
import os

DATA_PATH = os.getenv('DATA_PATH', '.')

def data_path(*args):
    # append the DATA_PATH as prefix to path join
    args = (DATA_PATH,) + args
    return os.path.join(*args)

def dict_apply_recur(d, fn, cols=None):
    # apply a function of some cols (or all if None) 
    # on all first-depth elements of d
    # if d has 'train' and 'test', we return the same dict
    # but where d['train']['X'] and d['test']['X'] are transformed
    # using fn
    # d : a dict
    # fn : function to apply
    return {k: dict_apply(v, fn, cols=cols) for k,v in d.items()}

def dict_apply(d, fn, cols=None):
    # apply a function on some cols (or all if None) and return
    # the changed dict
    # if d has 'X' and 'y' and cols is ['X'] then fn is applied on X
    # d : dict
    # fn : function to apply
    if not cols: cols = d.keys()
    d = d.copy()
    d.update({k: fn(d[k]) for k in cols})
    return d

def bufferize(iterator, buffer_size=128):
    # consume the iterator buffer_size times(at most), yield the consumed
    # elements one by one at once, repeat.
    buffer = []
    for data in iterator:
        buffer.append(data)
        if len(buffer) == buffer_size:
            for point in buffer:
                yield point
            buffer = []
    for point in buffer:
        yield point

def minibatch(iterator, batch_size=128):
    # consume the iterator batch_size times(at most), yield the consumed
    # elements as a list, unlike bufferize.
    buffer = []
    for data in iterator:
        buffer.append(data)
        if len(buffer) == batch_size:
            yield buffer
            buffer = []
    if buffer:
        yield buffer

def expand_dict(iterator):
    # iterator elements must be a dict.
    # if the iterator elements are of the form [{'X':..., 'y':...}, {'X':..., 'y':...}]
    # transform it to {'X': [...,...], 'y': [...., ...., ...]}
    for data in iterator:
        assert type(data) == list
        keys = data[0].keys()
        out = {}
        for k in keys:
            out[k] = map(lambda p:p[k], data)
        yield out

def as_iterator(data):
    # transform a data, which is a dict of the form of {'X':[...,...], 'y':[...,...]}, to an iterator
    # which yields elements of the form {'X': ..., 'y': ...}
    # all modalities (X, y, etc.) should be the same
    assert len(set(map(len, data.values()))) == 1
    k = data.keys()
    nb_examples = len(data[k[0]])
    iterator = imap(lambda i:{k: data[k][i] for k in data.keys()}, xrange(nb_examples))
    return iterator

def load_as_iterator(load_fn,):
    # a decorator to load functions to allow it to return iterators using as_iterator
    def fn(*args, **kwargs):
        data = load_fn(*args, **kwargs)
        return as_iterator(data)
    return fn

def ncycles(iterable, n):
    # does n cycle of an iterator by copying it in memory the first pass
    return chain.from_iterable(repeat(tuple(iterable), n))
