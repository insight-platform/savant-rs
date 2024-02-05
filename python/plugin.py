import savant_rs
import savant_py_plugin_sample
import time

frame = savant_rs.utils.gen_frame()
o = frame.get_object(0)

savant_py_plugin_sample.access_frame(frame)
savant_py_plugin_sample.access_object(o)
