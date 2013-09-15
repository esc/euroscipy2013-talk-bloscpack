#!/usr/bin/env python
# encoding: utf-8

""" Benchmarking utilities for comparing different Numpy array serialization
tools. """

import abc
import gc
import os
from time import time
from collections import OrderedDict as od

import progressbar as pbar
import numpy as np
from numpy.random import randn
import pandas as pd
import bloscpack as bp
import joblib as jb
import sh
import yaml


def noop():
    pass


def gen_results_filename():
    return 'results_' + str(int(time()))


def extract_config():

    def git_sha(base=''):
        try:
            return str(sh.git('rev-parse', 'HEAD', _cwd=base)).strip()
        except Exception:
            return 'NA'

    config = c = {}

    versions = v = {}
    v['bloscpack'] = bp.__version__
    v['numpy']     = np.__version__
    v['joblib']    = jb.__version__
    v['conda']     = str(sh.conda('--version', _tty_in=True)).strip()
    v['python']     = str(sh.python('--version', _tty_in=True)).strip()

    hashes = h = {}
    h['bloscpack'] = git_sha(os.path.dirname(bp.__file__))
    h['joblib'] = git_sha(jb.__path__[0])
    h['numpy'] = git_sha(np.__path__[0])
    h['benchmark'] = git_sha()

    c['uname'] = str(sh.uname('-a')).strip()
    c['hostname'] = str(sh.hostname()).strip()
    c['whoami'] = str(sh.whoami()).strip()
    c['date'] = str(sh.date()).strip()

    c['versions'] = versions
    c['hashes'] = hashes
    return config

def vtimeit(stmt, setup=noop, before=noop, after=noop, repeat=3, number=3):
    """ Specialised version of the timeit utility.

    Supports special operations. `setup` is performed once before each set.
    `before` is performed before each run but IS NOT included in the timing.
    `after` is performed after each run and IS included in the timing. Garbage
    collection is disables during runs.

    A `run` is a single a execution of the code to be benchmarked. A set is
    collection of runs. Usually, one perform a `repeat` of sets with `number`
    number of runs. Then, the average across runs is taken for each set and the
    minimum is selected as the final timing value.

    Parameters
    ----------

    stmt : callable
        the thing to benchmark
    setup : callable
        callable to be executed once before all runs
    before : callable
        callable to be executed once before every run
    after : callable
        callable to be executed once after every run
    repeat : int
        the number of sets
    number : int
        the number of runs in each set

    Returns
    -------
    results : ndarray
        2D array with `repeat` rows and `number` columns.
    """

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
    """ Drop linux file system caches. """
    bp.drop_caches()


def sync():
    """ Sync the linux file system buffers. """
    os.system('sync')


def make_simple_dataset(size):
    """ Make the dataset with low entropy. """
    return np.arange(size)


def make_complex_dataset(size):
    """ Make the dataset with medium entropy. """
    x = np.linspace(0, np.pi*2, 1e3)
    x = np.tile(x, size / len(x))
    assert len(x) == size
    y = np.sin(x)
    del x
    noise = np.random.randint(-1, 1, size) / 1e8
    it = y + noise
    del y
    del noise
    return it


def make_random_dataset(size):
    """ Make a bunch of random numbers. """
    return randn(size)


def reduce(result):
    """ Reduce the results array from a benchmark into a single value. """
    return result.mean(axis=1).min()


class AbstractRunner(object):
    """ Base class for a codec benchmark. """

    _metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def compress(self):
        """ Implement this to benchmark compression. """
        pass

    @abc.abstractmethod
    def decompress(self):
        """ Implement this to benchmark decompression. """
        pass

    def clean(self):
        """ Delete output, sync the buffers and clean the cache. """
        if os.path.isfile(self.storage):
            os.remove(self.storage)
        sync()
        drop_caches

    def deconfigure(self):
        """ Delete any storage internal to the instantiated object. """
        del self.ndarray
        gc.collect()

    def configure(self, ndarray, storage, level):
        """ Setup the input data, configure output and compression level. """
        self.ndarray = ndarray
        self.storage = os.path.join(storage, self.filename)
        self.level = level

    def ratio(self):
        """ Compute compression ratio. """
        return (float(os.path.getsize(self.storage)) /
                (self.ndarray.size * self.ndarray.dtype.itemsize))


class BloscpackRunner(AbstractRunner):
    """ Runner for Bloscpack.

    Files are generated without checksums, offsets and no preallocated chunks.
    """

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
                             blosc_args=blosc_args,
                             bloscpack_args=self.bloscpack_args)

    def decompress(self):
        it = bp.unpack_ndarray_file(self.storage)


class NPZRunner(AbstractRunner):
    """ Runner for NPZ. """

    def __init__(self):
        self.name = 'npz'
        self.filename = 'array.npz'

    def compress(self):
        np.savez_compressed(self.storage, self.ndarray)

    def decompress(self):
        it = np.load(self.storage)['arr_0']


class NPYRunner(AbstractRunner):
    """ Runner for NPY. """

    def __init__(self):
        self.name = 'npy'
        self.filename = 'array.npy'

    def compress(self):
        np.save(self.storage, self.ndarray)

    def decompress(self):
        it = np.load(self.storage)


class ZFileRunner(AbstractRunner):
    """ Runner for ZFile. """

    def __init__(self):
        self.name = 'npy'
        self.filename = 'array.npy'

    @property
    def storage_size(self):
        return os.path.getsize(self.storage) + os.path.getsize(self.real_data)

    @property
    def real_data(self):
        return self.storage + '_01.npy.z'

    def clean(self):
        if os.path.isfile(self.storage):
            os.remove(self.storage)
        if os.path.isfile(self.real_data):
            os.remove(self.real_data)
        sync()
        drop_caches

    def compress(self):
        jb.dump(self.ndarray, self.storage, compress=self.level, cache_size=0)

    def decompress(self):
        it = jb.load(self.storage)

    def ratio(self):
        return (float(self.storage_size) /
                (self.ndarray.size * self.ndarray.dtype.itemsize))

if __name__ == '__main__':

    result_file_name = gen_results_filename()
    conf = yaml.dump(extract_config(), default_flow_style=False)
    print conf
    conf_file = result_file_name + '.info.yaml'
    with open(conf_file, 'w') as fp:
        fp.write(conf)
    print 'config saved to: ' + conf_file

    ssd = '/tmp/bench'
    sd = '/mnt/sd/bench'

    for location in [ssd, sd]:
        if not os.path.isdir(location):
            raise Exception("Path: '%s' does not exist!" % location)

    dataset_sizes = od([('small', 1e4),
                        ('mid', 1e7),
                        ('large', 2e8),
                        ])
    storage_types = od([('ssd', ssd),
                        ('sd', sd),
                        ])
    entropy_types = od([('low', make_simple_dataset),
                        ('medium', make_complex_dataset),
                        ('high', make_random_dataset),
                        ])
    codecs = od([('bloscpack', BloscpackRunner()),
                 ('npz', NPZRunner()),
                 ('npy', NPYRunner()),
                 ('zfile', ZFileRunner()),
                 ])

    codec_levels = od([('bloscpack', [1, 3, 7, 9]),
                       ('npz', [1, ]),
                       ('npy', [0, ]),
                       ('zfile', [1, 3, 7]),
                       ])

    columns = ['size',
               'storage',
               'entropy',
               'codec',
               'level',
               'compress',
               'decompress',
               'dc_no_cache',
               'ratio',
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
    colum_values = od(zip(columns, zip(*sets)))
    colum_values['compress'] = np.empty(n)
    colum_values['decompress'] = np.empty(n)
    colum_values['dc_no_cache'] = np.empty(n)
    colum_values['ratio'] = np.empty(n)

    results = pd.DataFrame(colum_values)

    class Counter(pbar.Widget):
        """Displays the current count."""

        def update(self, pbar):
            try:
                return str(sets[pbar.currval-1])
            except IndexError:
                return ''

    widgets = ['Benchmark: ',
               pbar.Percentage(),
               ' ', Counter(),
               ' ',
               pbar.Bar(marker='-'),
               ' ',
               pbar.AdaptiveETA(),
               ' ',
               ]

    pbar = pbar.ProgressBar(widgets=widgets, maxval=n).start()

    for i, it in enumerate(sets):
        size, storage, entropy, codec, level = it

        if size == 'small':
            number = 10
            repeat = 4
        elif size == 'mid' or codec != 'zfile':
            number = 3
            repeat = 2
        elif size == 'large':
            number = 1
            repeat = 1
        else:
            raise RuntimeError("No such size: '%s'" % size)

        codec = codecs[codec]
        codec.configure(entropy_types[entropy](dataset_sizes[size]),
                        storage_types[storage], level)

        results['compress'][i] = reduce(vtimeit(codec.compress,
                                        setup=codec.compress,
                                        before=codec.clean, after=sync,
                                        number=number, repeat=repeat))
        results['ratio'][i] = codec.ratio()
        codec.deconfigure()
        results['decompress'][i] = reduce(vtimeit(codec.decompress,
                                                  setup=codec.decompress,
                                                  number=number,
                                                  repeat=repeat))
        results['dc_no_cache'][i] = reduce(vtimeit(codec.decompress,
                                                   before=drop_caches,
                                                   number=number,
                                                   repeat=repeat))

        codec.clean()
        pbar.update(i)

    pbar.finish()
    result_csv = result_file_name + '.csv'
    results.to_csv(result_csv)
    print 'results saved to: ' + result_csv
