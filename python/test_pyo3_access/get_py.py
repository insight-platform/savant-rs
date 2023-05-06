from timeit import default_timer as timer

import savant_rs

w = savant_rs.Wrapper(1)

t = timer()

for i in range(1_000_000):
    i = w.get()
    i.inc()

print(timer() - t)
print(w.get().val())
