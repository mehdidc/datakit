import os
from six.moves import map
from itertools import chain
from itertools import repeat
from functools import partial

DATA_PATH = os.getenv('DATA_PATH', '.')


def apply_to(fn, cols=None):
    def fn_(iterator, *args, **kwargs):
        iterator = map(partial(dict_apply, fn=partial(fn, *args, **kwargs), cols=cols), iterator)
        return iterator
    return fn_


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
    return {k: dict_apply(v, fn, cols=cols) for k, v in d.items()}


def dict_apply(d, fn, cols=None):
    # apply a function on some cols (or all if None) and return
    # the changed dict
    # if d has 'X' and 'y' and cols is ['X'] then fn is applied on X
    # d : dict
    # fn : function to apply
    if not cols:
        cols = d.keys()
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
    # considers the elements of iterator as arbitrary elements
    # and consume the iterator batch_size times(at most), yield the consumed
    # elements as a list, unlike bufferize.
    buffer = []
    for data in iterator:
        buffer.append(data)
        if len(buffer) == batch_size:
            yield buffer
            buffer = []
    if buffer:
        yield buffer


def minibatch_from_chunks(iterator, batch_size=128):
    # consider the elements of the iterator as a dict
    # where keys are modalities (e.g X and y) and values are iterable slicable
    # objects (e.g numpy array). Take at max batch_size elements from that objects
    # and iterate.
    for data in iterator:
        cols = list(data.keys())
        nb = len(data[cols[0]])
        for i in range(0, nb, batch_size):
            d = {}
            for c in cols:
                d[c] = data[c][i:i + batch_size]
            yield d


def expand_dict(iterator):
    # iterator elements must be a dict.
    # if the iterator elements are of the form [{'X': x1, 'y': y1}, {'X': x2, 'y': y2}]
    # transform it to iterator of {'X': [x1, x2], 'y': [y1, y2]}
    for data in iterator:
        assert type(data) == list
        keys = data[0].keys()
        out = {}
        for k in keys:
            out[k] = list(map(lambda p: p[k], data))
        yield out


def as_iterator(data):
    # transform a data, which is a dict of the form of {'X':[x1, x2], 'y':[y1, y2]}, to an iterator
    # which yields elements of the form {'X': x1, 'y': y1} then {'X': x2, 'y': y2}
    assert len(set(map(len, data.values()))) == 1
    k = list(data.keys())
    nb_examples = len(data[k[0]])
    iterator = map(lambda i: {k: data[k][i] for k in data.keys()}, range(nb_examples))
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


def format_with_env(s):
    # format a string by supplying environment variables
    return s.format(**os.environ)
