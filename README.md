datakit is a simple data loader and preprocessing tool.
The goal is to load raw data (e.g npy, list of images, hdf5, etc.), apply a series of transformations
and return an iterator, for which each element ca typicall be a minibatch sampled from the data.
