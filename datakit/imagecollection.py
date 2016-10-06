from itertools import imap
from itertools import cycle
from functools import partial

from skimage.io import imread

from helpers import bufferize, dict_apply, ncycles, minibatch, expand_dict

from constants import *

def load(filelist):
    X = build_image_loader_iterator(filelist)
    X = list(X)
    return {TRAIN: {'X': X}}

def build_image_loader_iterator(filelist):
    iterator = imap(imread, filelist)
    return iterator

def load_as_iterator(filelist):
    iterator = build_image_loader_iterator(filelist)
    iterator = imap(lambda value: {'X': value}, iterator)
    return iterator
