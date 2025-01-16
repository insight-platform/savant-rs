from savant_rs.utils import AtomicCounter

base = 100000
counter = AtomicCounter(base)

print(counter.next)
print(counter.next)
print(counter.get)
counter.set(1000)
