import numpy as np
from savant_rs.utils import *
from timeit import default_timer as timer

num = 10_000
dims = (1024, 1)


def bench(dtype, dims, num):
    t = timer()
    v = np.zeros(dims, dtype=dtype)
    m = None
    for _ in range(num):
        m = ndarray_to_matrix(v)
    print(f"NP {dtype} > NALGEBRA {dtype}: {num} Time:", timer() - t)

    t = timer()
    for _ in range(num):
        v = matrix_to_ndarray(m)

    print(f"NALGEBRA {dtype} > NP {dtype}: {num} Time:", timer() - t)


for dt in ['uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32', 'uint64', 'int64', 'float32', 'float64']:
    bench(dt, dims, num)
