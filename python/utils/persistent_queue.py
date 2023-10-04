import time
from savant_rs.utils import PersistentQueue

# Create a queue that will persist to disk

q = PersistentQueue('/tmp/queue')

start = time.time()
buf = bytes(256*1024)

print("begin writing")
for i in range(1000):
    q.try_push(buf)

print("begin reading")
for i in range(1000):
    q.try_pop()

end = time.time()

print("Time taken: %f" % (end - start))

