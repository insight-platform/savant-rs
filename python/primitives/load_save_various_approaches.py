import ctypes

from savant_rs.utils import gen_frame
from savant_rs.utils.serialization import load_message, load_message_from_bytes, \
    load_message_from_bytebuffer, save_message, save_message_to_bytes, save_message_to_bytebuffer, Message
from timeit import default_timer as timer

from savant_rs.logging import LogLevel, set_log_level
set_log_level(LogLevel.Trace)

f = gen_frame()
m = Message.video_frame(f)
t = timer()
for _ in range(1_00):
    s = save_message(m)
    new_m = load_message(s)
    assert new_m.is_video_frame()

print("Regular Save/Load", timer() - t)

t = timer()
for _ in range(1_00):
    s = save_message_to_bytebuffer(m, with_hash=False)
    new_m = load_message_from_bytebuffer(s)
    assert new_m.is_video_frame()

print("ByteBuffer (no hash) Save/Load", timer() - t)

t = timer()
for _ in range(1_00):
    s = save_message_to_bytebuffer(m, with_hash=True)
    new_m = load_message_from_bytebuffer(s)
    assert new_m.is_video_frame()

print("ByteBuffer (with hash) Save/Load", timer() - t)

t = timer()
for _ in range(1_00):
    s = save_message_to_bytes(m)
    new_m = load_message_from_bytes(s)
    assert new_m.is_video_frame()

print("Python bytes Save/Load", timer() - t)
