from multiprocessing import Process
from time import time
import zmq

from savant_rs.logging import LogLevel, set_log_level
set_log_level(LogLevel.Trace)

from savant_rs.utils import gen_frame
from savant_rs.utils.serialization import load_message_from_bytes, save_message_to_bytes, Message
from savant_rs.primitives import VideoFrameUpdate, VideoObjectUpdateCollisionResolutionPolicy, \
    AttributeUpdateCollisionResolutionPolicy


socket_name = "ipc://test_hello"


def server():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.connect(socket_name)
    while True:
        message = socket.recv()
        if message == b'end':
            print("Received end")
            break

        _ = load_message_from_bytes(message)

        update = VideoFrameUpdate()
        update.object_collision_resolution_policy = VideoObjectUpdateCollisionResolutionPolicy.add_foreign_objects()
        update.attribute_collision_resolution_policy = AttributeUpdateCollisionResolutionPolicy.replace_with_foreign()

        m = Message.video_frame_update(update)
        binary = save_message_to_bytes(m)
        socket.send(binary)


frame = gen_frame()
p1 = Process(target=server)
p1.start()

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.bind(socket_name)

start = time()
wait_time = 0
for _ in range(1):
    m = Message.video_frame(frame)
    s = save_message_to_bytes(m)
    socket.send(s)
    wait = time()
    m = socket.recv()
    wait_time += (time() - wait)
    m = load_message_from_bytes(m)
    assert m.is_video_frame_update()

print("Time taken", time() - start, wait_time)
socket.send(b'end')
p1.join()
