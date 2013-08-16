import gc
import io
import os
from time import time
import numpy as np


import bloscpack as bp

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

dataset_size = [1e4]
ssd = '/tmp/bench'
storage_type = [ssd]


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

class BloscpackRunner(object):

    def configure(self, ndarray, storage):
        self.ndarray = ndarray
        self.storage = os.path.join(storage, 'array.blp')

    def compress(self):
        bp.pack_ndarray_file(self.ndarray, self.storage)

    def decompress(self):
        it = bp.unpack_ndarray_file(self.storage)


bp_runner = BloscpackRunner()

bp_runner.configure(make_simple_dataset(dataset_size[0]), storage_type[0])
print vtimeit(bp_runner.compress, setup=bp_runner.compress, before=drop_caches, after=sync)
print vtimeit(bp_runner.decompress, setup=bp_runner.decompress)
print vtimeit(bp_runner.decompress, before=drop_caches)

bp_runner.configure(make_complex_dataset(dataset_size[0]), storage_type[0])
print vtimeit(bp_runner.compress, before=drop_caches, after=sync)
print vtimeit(bp_runner.decompress, setup=bp_runner.decompress)
print vtimeit(bp_runner.decompress, before=drop_caches)
