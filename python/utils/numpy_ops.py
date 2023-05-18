import numpy as np
from savant_rs.utils import *
from timeit import default_timer as timer

num = 10_000
dims = (1024, 1)


def bench_matrix(dtype, dims, num):
    t = timer()
    v = np.zeros(dims, dtype=dtype)
    m = None
    for _ in range(num):
        m = np_to_matrix(v)
    print(f"NP {dtype} > NALGEBRA {dtype}: {num} Time:", timer() - t)

    t = timer()
    for _ in range(num):
        v = matrix_to_np(m)

    print(f"NALGEBRA {dtype} > NP {dtype}: {num} Time:", timer() - t)


def bench_ndarray(dtype, dims, num):
    t = timer()
    v = np.zeros(dims, dtype=dtype)
    m = None
    for _ in range(num):
        m = np_to_ndarray(v)
    print(f"NP {dtype} > NDARRAY {dtype}: {num} Time:", timer() - t)

    t = timer()
    for _ in range(num):
        v = ndarray_to_np(m)

    print(f"NDARRAY {dtype} > NP {dtype}: {num} Time:", timer() - t)


print("Bench Nalgebra Matrix")
for dt in ['uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32', 'uint64', 'int64', 'float32', 'float64']:
    bench_matrix(dt, dims, num)

print("Bench NDArray")
for dt in ['uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32', 'uint64', 'int64', 'float32', 'float64']:
    bench_ndarray(dt, dims, num)


arr = np.zeros((3, 4), dtype='float32')
print(np_to_ndarray(arr))
