from timeit import default_timer as timer


class Internal:
    def __init__(self, val):
        self.val = val

    def inc(self):
        self.val += 1


class Wrapper:
    def __init__(self, val):
        self.val = Internal(val)

    def get(self):
        return self.val


w = Wrapper(1)
t = timer()

for _ in range(1_000_000):
    i = w.get()
    i.inc()

print(timer() - t)
i = w.get()
print(i.val)
