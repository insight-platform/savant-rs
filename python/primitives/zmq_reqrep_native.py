from threading import Thread
from time import time

from savant_rs.utils import gen_frame
from savant_rs.utils.serialization import Message
from savant_rs.zmq import WriterConfigBuilder, ReaderConfigBuilder, Writer, Reader

socket_name = "ipc:///tmp/test_hello"

NUMBER = 1000
BLOCK_SIZE = 1024 * 1024

writer_config = WriterConfigBuilder("req+bind:" + socket_name).build()
writer = Writer(writer_config)
writer.start()


def server():
    reader_config = ReaderConfigBuilder("rep+connect:" + socket_name).build()
    reader = Reader(reader_config)
    reader.start()
    for _ in range(NUMBER):
        m = reader.receive()
        assert len(m.data(0)) == BLOCK_SIZE

frame = gen_frame()
p1 = Thread(target=server)
p1.start()

buf = bytes(BLOCK_SIZE)

start = time()
wait_time = 0
for _ in range(NUMBER):
    m = Message.video_frame(frame)
    wait = time()
    writer.send_message("topic", m, buf)
    wait_time += (time() - wait)

print("Time taken", time() - start, wait_time)
p1.join()
