import abc
import gc
import io
import itertools
import os
import sys
from time import time
from collections import OrderedDict as od

import progressbar as pbar
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

    def configure(self, ndarray, storage, level):
        self.ndarray = ndarray
        self.storage = os.path.join(storage, self.filename)
        self.level = level

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
        blosc_args = self.blosc_args.copy()
        blosc_args['clevel'] = self.level
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
        jb.dump(self.ndarray, self.storage, compress=self.level, cache_size=0)

    def decompress(self):
        it = jb.load(self.storage)


ssd = '/tmp/bench'
sd = '/mnt/sd/bench'
dataset_sizes = od([('small', 1e4),])
                  #  ('mid', 1e7),
                  #  ('large', 2e8),
                  #  ])
storage_types = od([('ssd', ssd)])
entropy_types = od([('low', make_simple_dataset),
                    ('medium', make_complex_dataset),])
                   # ('high', make_random_dataset),
                   # ])
codecs = od([('bloscpack', BloscpackRunner()),
             ('npz', NPZRunner()),
             ('npy', NPYRunner()),
             ('zfile', ZFileRunner()),
             ])

codec_levels = od([('bloscpack', [1, 3, 7, 9]),
                   ('npz', [1, ]),
                   ('npy', [0, ]),
                   ('zfile', [1, 3, 7, 9]),
                   ])

columns = ['size',
           'storage',
           'entropy',
           'codec',
           'level',
           'compress',
           'decompress',
           'dc_no_cache',
           'ratio'
           ]

sets = []
# can't use itertools.product, because level depends on codec
for size in dataset_sizes:
    for type_ in storage_types:
        for entropy in entropy_types:
            for codec in codecs:
                for level in codec_levels[codec]:
                    sets.append((size, type_, entropy, codec, level))

n = len(sets)
colum_values =od(zip(columns, zip(*sets)))
colum_values['compress'] = np.empty(n)
colum_values['decompress'] = np.empty(n)
colum_values['dc_no_cache'] = np.empty(n)
colum_values['ratio'] = np.empty(n)

results = pd.DataFrame(colum_values)

widgets = ['Benchmark: ', pbar.Percentage(), ' ', pbar.Bar(marker='-'),
           ' ', pbar.AdaptiveETA(), ' ', ]
pbar = pbar.ProgressBar(widgets=widgets, maxval=n).start()

for i, it in enumerate(sets):
    size, storage, entropy, codec, level = it
    codec = codecs[codec]
    codec.configure(entropy_types[entropy](dataset_sizes[size]),
                    storage_types[storage], level)
    results['compress'][i] = reduce(vtimeit(codec.compress, setup=codec.compress,
                 before=codec.clean, after=sync))
    results['ratio'][i] = codec.ratio()
    codec.deconfigure()
    results['decompress'][i] = reduce(vtimeit(codec.decompress, setup=codec.decompress))
    results['dc_no_cache'][i] = reduce(vtimeit(codec.decompress, before=drop_caches))
    pbar.update(i)

pbar.finish()
print results
results.to_csv('results.csv')
