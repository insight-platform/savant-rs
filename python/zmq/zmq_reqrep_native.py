from time import time

from savant_rs.zmq import ReaderConfigBuilder, BlockingReader

socket_name = "tcp://192.168.1.135:3333"

NUMBER = 1000
BLOCK_SIZE = 128 * 1024


def server():
    reader_config = ReaderConfigBuilder("router+bind:" + socket_name).build()
    reader = BlockingReader(reader_config)
    reader.start()
    wait_time = 0
    for _ in range(NUMBER):
        wait = time()
        m = reader.receive()
        wait_time += (time() - wait)
        assert len(m.data(0)) == BLOCK_SIZE
        print(m.topic)
    print("Reader time awaited", wait_time)


server()
