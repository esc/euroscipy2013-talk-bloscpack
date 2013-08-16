import gc
from time import time
import numpy as np


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
            toc = time()
            result[i, j] = toc - tic

            if gcold:
                gc.enable()

    return result


