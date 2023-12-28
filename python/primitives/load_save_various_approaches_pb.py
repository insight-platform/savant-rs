from timeit import default_timer as timer

from savant_rs.logging import LogLevel, set_log_level
from savant_rs.utils import gen_frame
from savant_rs.utils.serialization import load_pb_message, load_pb_message_from_bytes, \
    load_pb_message_from_bytebuffer, save_pb_message, save_pb_message_to_bytes, save_pb_message_to_bytebuffer, Message

f = gen_frame()
m = Message.video_frame(f)
t = timer()
for _ in range(1_00):
    s = save_pb_message(m)
    new_m = load_pb_message(s)
    assert new_m.is_video_frame()

print("Regular Save/Load", timer() - t)

t = timer()
for _ in range(1_00):
    s = save_pb_message_to_bytebuffer(m, with_hash=False)
    new_m = load_pb_message_from_bytebuffer(s)
    assert new_m.is_video_frame()

print("ByteBuffer (no hash) Save/Load", timer() - t)

t = timer()
for _ in range(1_00):
    s = save_pb_message_to_bytebuffer(m, with_hash=True)
    new_m = load_pb_message_from_bytebuffer(s)
    assert new_m.is_video_frame()

print("ByteBuffer (with hash) Save/Load", timer() - t)

t = timer()
for _ in range(1_00):
    s = save_pb_message_to_bytes(m)
    new_m = load_pb_message_from_bytes(s)
    assert new_m.is_video_frame()

print("Python bytes Save/Load", timer() - t)
