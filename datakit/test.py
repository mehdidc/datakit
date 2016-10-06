from itertools import repeat
from itertools import imap
from itertools import cycle
from functools import partial

from helpers import bufferize, dict_apply, ncycles, minibatch, expand_dict

import mnist
import cifar
import imagecollection

from constants import *


def test_mnist():
    data = mnist.load()

def test_cifar():
    data = cifar.load()

def test_image_collection():
    from skimage.transform import resize
    import glob
    import numpy as np
    filelist = glob.glob('/home/mcherti/work/data/lfw/img/**/*.jpg')
    filelist = filelist[0:100]
    iterator = imagecollection.load_as_iterator(filelist)
    resize_ = partial(resize, output_shape=(28, 28), preserve_range=True)
    iterator = imap(partial(dict_apply, fn=resize_, cols=['X']), iterator)
    iterator = minibatch(iterator, batch_size=10)
    iterator = expand_dict(iterator)
    iterator = imap(partial(dict_apply, fn=np.array, cols=['X']), iterator)
    for p in iterator:
        pass

if __name__ == '__main__':
    pass
