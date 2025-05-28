#!/usr/bin/env python3
import argparse
import sys

from savant_rs.logging import LogLevel, set_log_level
from savant_rs.utils import gen_frame
from savant_rs.utils.serialization import Message
from savant_rs.zmq import WriterConfigBuilder, BlockingWriter, WriterResultSuccess

set_log_level(LogLevel.Info)


def main():
    parser = argparse.ArgumentParser(description="ZMQ Message Producer")
    parser.add_argument(
        "--socket",
        required=True,
        help="ZMQ socket URI (e.g. dealer+connect:tcp://127.0.0.1:6666)",
    )
    parser.add_argument(
        "--count", type=int, default=1000, help="Number of messages to send"
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=128*1024,
        help="Size of each message in bytes",
    )
    args = parser.parse_args()

    # Generate test data
    frame = gen_frame()
    buf = bytes(args.block_size)

    # Configure and start writer
    writer_config = WriterConfigBuilder(args.socket).build()
    writer = BlockingWriter(writer_config)
    writer.start()

    try:
        for i in range(args.count):
            m = Message.video_frame(frame)
            res = writer.send_message("topic", m, buf)
            if res.__class__ != WriterResultSuccess:
                print("Failed to send message")
                continue
            i += 1
            if (i + 1) % 100 == 0:
                print(f"Sent {i + 1} messages...")

        print(f"Sent {args.count} messages")

    except KeyboardInterrupt:
        print("\nProducer interrupted by user")


if __name__ == "__main__":
    main()
