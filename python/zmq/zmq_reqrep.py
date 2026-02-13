from threading import Thread
from time import time

import zmq
from savant_rs.utils import gen_frame
from savant_rs.utils.serialization import (
    Message,
    load_message_from_bytes,
    save_message_to_bytes,
)

socket_name = "ipc:///tmp/test_hello"

NUMBER = 1000
BLOCK_SIZE = 1024 * 1024


def server():
    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    socket.connect(socket_name)
    while True:
        message = socket.recv_multipart()
        if message[1] == b"end":
            print("Received end")
            break

        _ = load_message_from_bytes(message[1])


frame = gen_frame()
p1 = Thread(target=server)
p1.start()

context = zmq.Context()
socket = context.socket(zmq.DEALER)
socket.bind(socket_name)

buf_1024b = bytes(BLOCK_SIZE)

start = time()
wait_time = 0
m = Message.video_frame(frame)
for _ in range(NUMBER):
    s = save_message_to_bytes(m)
    socket.send_multipart([s, buf_1024b])
    wait = time()
    wait_time += time() - wait

print("Time taken", time() - start, wait_time)
socket.send_multipart([b"end"])
p1.join()
