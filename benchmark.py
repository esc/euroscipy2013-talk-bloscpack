import abc
import gc
import io
import itertools
import os
from time import time
from collections import OrderedDict as od

import numpy as np
from numpy.random import randn
import pandas as pd
import bloscpack as bp
import joblib as jb


def noop():
    pass


def vtimeit(stmt, setup=noop, before=noop, after=noop, repeat=3, number=3):
    result = np.empty((repeat, number))
    for i in range(repeat):
        setup()
        for j in range(number):
            before()

            gcold = gc.isenabled()
            gc.disable()

            tic = time()
            stmt()
            after()
            toc = time()
            result[i, j] = toc - tic


            if gcold:
                gc.enable()

    return result


def drop_caches():
    bp.drop_caches()


def sync():
    os.system('sync')


def make_simple_dataset(size):
    return np.arange(size)


def make_complex_dataset(size):
    x = np.linspace(0, np.pi*16, size)
    y = np.sin(x)
    del x
    noise = np.random.randint(-1, 1, size) / 1e8
    it = y + noise
    del y
    del noise
    return it


def make_random_dataset(size):
    return randn(size)


def reduce(result):
    return result.mean(axis=1).min()


class AbstractRunner(object):

    _metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def compress(self):
        pass

    @abc.abstractmethod
    def decompress(self):
        pass

    def clean(self):
        if os.path.isfile(self.storage):
            os.remove(self.storage)
        sync()
        drop_caches

    def deconfigure(self):
        del self.ndarray
        gc.collect()

    def configure(self, ndarray, storage):
        self.ndarray = ndarray
        self.storage = os.path.join(storage, self.filename)

    def ratio(self):
        return (float(os.path.getsize(self.storage)) /
                (self.ndarray.size * self.ndarray.dtype.itemsize))


class BloscpackRunner(AbstractRunner):

    def __init__(self):
        self.name = 'bloscpack'
        self.blosc_args = bp.DEFAULT_BLOSC_ARGS
        self.bloscpack_args = bp.DEFAULT_BLOSCPACK_ARGS
        self.bloscpack_args = {'offsets': False,
                               'checksum': 'None',
                               'max_app_chunks': 0}
        self.filename = 'array.blp'

    def compress(self):
        bp.pack_ndarray_file(self.ndarray, self.storage,
                             blosc_args=self.blosc_args,
                             bloscpack_args=self.bloscpack_args)

    def decompress(self):
        it = bp.unpack_ndarray_file(self.storage)


class NPZRunner(AbstractRunner):

    def __init__(self):
        self.name = 'npz'
        self.filename = 'array.npz'

    def compress(self):
        np.savez_compressed(self.storage, self.ndarray)

    def decompress(self):
        it = np.load(self.storage)['arr_0']


class NPYRunner(AbstractRunner):

    def __init__(self):
        self.name = 'npy'
        self.filename = 'array.npy'

    def compress(self):
        np.save(self.storage, self.ndarray)

    def decompress(self):
        it = np.load(self.storage)


class ZFileRunner(AbstractRunner):

    def __init__(self):
        self.name = 'npy'
        self.filename = 'array.npy'

    def compress(self):
        jb.dump(self.ndarray, self.storage, compress=3, cache_size=0)

    def decompress(self):
        it = jb.load(self.storage)



ssd = '/tmp/bench'
sd = '/mnt/sd/bench'
dataset_sizes = od([('small', 1e4),
                    ('medium', 1e7),
                    ('large', 2e8),
                    ])
storage_types = od([('ssd', ssd)])
entropy_types = od([('low', make_simple_dataset),
                    ('medium', make_complex_dataset),
                    ('high', make_random_dataset),
                    ])
codecs = {'bloscpack': BloscpackRunner()}

columns = ['size',
           'entropy',
           'storage',
           'codec',
           'level',
           'compress',
           'decompress',
           'decompress w/o cache'
           'ratio'
           ]

sets = itertools.product(dataset_sizes, storage_types, entropy_types, codecs)

#results = pd.DataFrame()

for i, it in enumerate(sets):
    print it
    size, storage, entropy, codec = it
    codec = codecs[codec]
    codec.configure(entropy_types[entropy](dataset_sizes[size]),
                    storage_types[storage])
    print reduce(vtimeit(codec.compress, setup=codec.compress,
                 before=codec.clean, after=sync))
    print codec.ratio()
    codec.deconfigure()
    print reduce(vtimeit(codec.decompress, setup=codec.decompress))
    print reduce(vtimeit(codec.decompress, before=drop_caches))
