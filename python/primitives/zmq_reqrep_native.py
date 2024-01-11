from threading import Thread
from time import time

from savant_rs.utils import gen_frame
from savant_rs.utils.serialization import Message
from savant_rs.zmq import WriterConfigBuilder, ReaderConfigBuilder, BlockingWriter, BlockingReader

socket_name = "ipc:///tmp/test_hello"

NUMBER = 1000
BLOCK_SIZE = 1024 * 1024

writer_config = WriterConfigBuilder("dealer+bind:" + socket_name).build()
writer = BlockingWriter(writer_config)
writer.start()


def server():
    reader_config = ReaderConfigBuilder("router+connect:" + socket_name).build()
    reader = BlockingReader(reader_config)
    reader.start()
    wait_time = 0
    for _ in range(NUMBER):
        wait = time()
        m = reader.receive()
        wait_time += (time() - wait)
        assert len(m.data(0)) == BLOCK_SIZE
    print("Reader time awaited", wait_time)


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

print("Writer time taken", time() - start, "awaited", wait_time)
p1.join()
