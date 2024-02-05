import savant_rs
import savant_py_plugin_sample
import time

frame = savant_rs.utils.gen_frame()
o = frame.get_object(0)

print(frame.memory_handle)
print(hash(frame))

savant_py_plugin_sample.access_frame(frame, no_gil = True)
savant_py_plugin_sample.access_object(o, no_gil = True)

while True:
    time.sleep(1)
