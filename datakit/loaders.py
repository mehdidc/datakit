import numpy as np
import h5py

from .helpers import format_with_env

_load_numpy_cache = {}


def pipeline_load_numpy(iterator, filename,
                        cols=['X'],
                        start=0, nb=None, shuffle=False,
                        chunk_size=128,
                        cache=True,
                        cache_dict=_load_numpy_cache,
                        random_state=None):
    """
    Operator to load npy or npz files

    Parameters
    ----------

    filename : str
        filename to load
    cols : list of str
        columns to retrieve from the npy file
    start : int(default=0)
        starting index of the data
    nb : int(default=None)
        the size of the data to read.
        if None, take everything starting
        from start.
    shuffle : bool(default=False)
        whether to shuffle the data
    cache : bool(default=True)
        whether to store loaded data in a cache for next calls
    cache_dict : dict
        where to store the cache
    random_state : int(default=None)
    """
    rng = np.random.RandomState(random_state)

    if cache and filename in _load_numpy_cache:
        data = _load_numpy_cache[filename]
    else:
        filename = format_with_env(filename)
        data = np.load(filename)
        if cache:
            _load_numpy_cache[filename] = data

    if shuffle:
        indices = np.arange(len(data[cols[0]]))
        rng.shuffle(indices)
        data_shuffled = {}
        for c in cols:
            data_shuffled[c] = data[c][indices]
        data = data_shuffled

    if nb is None:
        nb = len(data[data.keys()[0]])

    def iter_func():
        for i in range(start, start + nb, chunk_size):
            d = {}
            for c in cols:
                d[c] = data[c][i:i + chunk_size]
            yield d
    return iter_func()


def pipeline_load_hdf5(iterator, filename,
                       cols=['X'],
                       start=0, nb=None, chunk_size=128):
    """
    Operator to load hdf5 files

    Paramters
    ---------

    filename : str
        filename to load
    cols : list of str
        columns to retrieve from the npy file
    start : int(default=0)
        starting index of the data
    nb : int(default=None)
        the size of the data to read.
        if None, take everything starting
        from start.
    buffer_size : int(default=128)
        read buffer_size rows each time from the file
    random_state : int(default=None)

    """

    filename = format_with_env(filename)
    hf = h5py.File(filename, 'r')
    if nb is None:
        nb = hf[cols[0]].shape[0]

    def iter_func():
        for i in range(start, start + nb, chunk_size):
            d = {}
            for c in cols:
                d[c] = hf[c][i:i + chunk_size]
            yield d
    return iter_func()


operators = {
    'load_numpy': pipeline_load_numpy,
    'load_hdf5': pipeline_load_hdf5
}
