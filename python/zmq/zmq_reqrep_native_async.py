import asyncio
from time import time

from savant_rs.utils import gen_frame
from savant_rs.utils.serialization import Message
from savant_rs.zmq import (
    NonBlockingReader,
    NonBlockingWriter,
    ReaderConfigBuilder,
    WriterConfigBuilder,
)

socket_name = "ipc:///tmp/test_hello"

NUMBER = 1000
BLOCK_SIZE = 1024 * 1024


async def reader():
    reader_config = ReaderConfigBuilder("rep+connect:" + socket_name).build()
    reader = NonBlockingReader(reader_config, 100)
    reader.start()
    counter = NUMBER
    while counter > 0:
        m = reader.try_receive()
        if m is None:
            await asyncio.sleep(0)
        else:
            assert len(m.data(0)) == BLOCK_SIZE

            if counter % 1000 == 0:
                print(
                    "Read counter",
                    counter,
                    ", enqueued results",
                    reader.enqueued_results(),
                )

            counter -= 1


async def writer():
    writer_config = WriterConfigBuilder("req+bind:" + socket_name).build()
    writer = NonBlockingWriter(writer_config, 100)
    writer.start()

    frame = gen_frame()
    buf = bytes(BLOCK_SIZE)
    start = time()
    wait_time = 0
    counter = NUMBER
    while counter > 0:
        m = Message.video_frame(frame)
        wait = time()
        response = writer.send_message("topic", m, buf)
        while response.try_get() is None:
            await asyncio.sleep(0)

        if counter % 1000 == 0:
            print("Write counter", counter)

        counter -= 1
        wait_time += time() - wait

    print("Time taken", time() - start, wait_time)


async def run_system():
    await asyncio.gather(reader(), writer())


loop = asyncio.get_event_loop()
try:
    loop.run_until_complete(run_system())
finally:
    loop.run_until_complete(
        loop.shutdown_asyncgens()
    )  # see: https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.loop.shutdown_asyncgens
    loop.close()
