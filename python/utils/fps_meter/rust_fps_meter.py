from savant_rs.utils import FpsMeter
from timeit import default_timer as timer

from savant_rs.logging import LogLevel, set_log_level
set_log_level(LogLevel.Trace)

m = FpsMeter.time_based(5)

t = timer()

res = None
for i in range(10_000_000):
    res = m(i)

fps = FpsMeter.fps(*res)
print(fps)

m = FpsMeter.message(*res)
print(m)

print(timer() - t)
