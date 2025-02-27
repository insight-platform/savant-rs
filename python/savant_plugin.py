import savant_rs
import savant_plugin_sample

frame = savant_rs.utils.gen_frame()
o = frame.get_object(0)
print(o)

savant_plugin_sample.access_frame(frame)
savant_plugin_sample.access_object(o)
