from multiprocessing import Process
from time import time

import zmq

from savant_rs.utils import gen_frame
from savant_rs.utils.serialization import load_message_from_bytes, save_message_to_bytes, Message
from savant_rs.primitives import VideoFrameUpdate, ObjectUpdatePolicy, AttributeUpdatePolicy

socket_name = "ipc:///tmp/test_hello"


def server():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.connect(socket_name)
    while True:
        message = socket.recv_multipart()
        if message[0] == b'end':
            print("Received end")
            break

        _ = load_message_from_bytes(message[0])
        socket.send(b'OK')

frame = gen_frame()
p1 = Process(target=server)
p1.start()

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.bind(socket_name)

buf_1024b = bytes(1024*1024)

start = time()
wait_time = 0
for _ in range(1000):
    m = Message.video_frame(frame)
    s = save_message_to_bytes(m)
    socket.send_multipart([s, buf_1024b])
    wait = time()
    m = socket.recv()
    wait_time += (time() - wait)
    assert m == b'OK'

print("Time taken", time() - start, wait_time)
socket.send_multipart([b'end'])
p1.join()
