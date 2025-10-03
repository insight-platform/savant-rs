#!/usr/bin/env python3
import argparse
import sys
from time import time

from savant_rs.logging import LogLevel, set_log_level
from savant_rs.zmq import ReaderConfigBuilder, BlockingReader, ReaderResultTimeout
from time import time

set_log_level(LogLevel.Info)


def main():
    parser = argparse.ArgumentParser(description="ZMQ Message Consumer")
    parser.add_argument(
        "--socket",
        required=True,
        help="ZMQ socket URI (e.g. router+bind:tcp://127.0.0.1:6666)",
    )
    parser.add_argument(
        "--count", type=int, default=1000, help="Number of messages to receive"
    )
    args = parser.parse_args()

    # Configure and start reader
    reader_config = ReaderConfigBuilder(args.socket).build()
    reader = BlockingReader(reader_config)
    reader.start()

    i = 0
    now = time()
    try:
        while i < args.count:
            m = reader.receive()
            if m.__class__ == ReaderResultTimeout:
                continue
            i += 1
            if i % 1000 == 0:
                print(f"Received {i} messages")

        print(f"Time taken: {time() - now}s")

    except KeyboardInterrupt:
        print("\nConsumer interrupted by user")


if __name__ == "__main__":
    main()
