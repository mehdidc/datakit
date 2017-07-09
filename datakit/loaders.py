import numpy as np
import h5py

from .helpers import format_with_env

_load_numpy_cache = {}


def pipeline_load_numpy(iterator, filename,
                        cols=['X'],
                        start=0, nb=None, shuffle=False,
                        chunk_size=None,
                        cache=True,
                        cache_dict=_load_numpy_cache,
                        random_state=None):
    """
    Operator to load npy or npz files

    Parameters
    ----------

    iterator : starting iterator
        not used, it is just to follow the API
        of pipelines, which take an iterator and
        return a transformed iterator. But load_numpy
        should be the first step in the pipeline, so
        at that point iterator=None.
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
    chunk_size : int(default=None)
        size of the chunk (nb of examples) to load into memory each time.
        (it does not have to be necessarily the size of the mini-batch,
         these two quantities are independent). if it is None, the size
         of the chunk is the size of the data (load all the data into memory).
    cache : bool(default=True)
        whether to store loaded data in a cache for next calls
    cache_dict : dict
        where to store the cache
    random_state : int(default=None)
        used to create RandomState object to shuffle 
        the data with in case shuffle is True.
    """
    rng = np.random.RandomState(random_state)
    # retrieve from cache if it exists
    if cache and filename in _load_numpy_cache:
        data = _load_numpy_cache[filename]
    else:
        # normal behavior : when the object does not exist in the cache
        filename = format_with_env(filename)
        data = np.load(filename)
        if cache:
            d = {}
            for col in data.keys():
                d[col] = data[col][:]
            _load_numpy_cache[filename] = d
    if nb is None:
        nb = len(data[list(data.keys())[0]])

    if shuffle:
        #WARNING
        # when shuffled is True, we first shuffle the whole data.
        # Then, a slice defined by (start, start+nb) is
        # selected.
        indices = np.arange(len(data[cols[0]]))
        rng.shuffle(indices)
        data_shuffled = {}
        for c in cols:
            data_shuffled[c] = data[c][indices][start:start + nb]
        data = data_shuffled
    else:
        d = {}
        for c in cols:
            d[c] = data[c][start:start+nb]
        data = d
    if chunk_size is None:
        chunk_size = nb
    def iter_func():
        for i in range(0, nb, chunk_size):
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
    
    iterator : starting iterator
        not used, it is just to follow the API
        of pipelines, which take an iterator and
        return a transformed iterator. But load_hdf5
        should be the first step in the pipeline, so
        at that point iterator=None.
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
    chunk_size : int(default=128)
        read chunk_size_size rows each time from the file
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
