try:
    from itertools import imap
except ImportError:
    imap = map

from skimage.io import imread
from .constants import TRAIN

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
