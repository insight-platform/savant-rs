import numpy as np
from savant_rs.utils import *
from timeit import default_timer as timer

t = timer()
v = np.zeros((1024, 1), dtype='float')
m = None
for _ in range(100_000):
    m = ndarray_to_matrix(v)

print("NP64>NALGEBRA 100K Time:", timer() - t)

t = timer()
for _ in range(100_000):
    v = matrix_to_ndarray(m)

print("NALGEBRA>NP64 100K Time:", timer() - t)


t = timer()
v = np.zeros((1024, 1), dtype='float32')
m = None
for _ in range(100_000):
    m = ndarray_to_matrix(v)
print("NP32>NALGEBRA 100K Time:", timer() - t)

t = timer()
for _ in range(100_000):
    v = matrix_to_ndarray(m)

print("NALGEBRA>NP32 100K Time:", timer() - t)


v = np.zeros((4, 8), dtype='float')
m = ndarray_to_matrix(v)
print(m)