import time
import threading
import numpy as np
from savant_rs.logging import log, LogLevel, set_log_level
from savant_rs.primitives import AttributeValue, Attribute
from savant_rs.webserver.kvs import set_attributes

import shared_state

shared_state.__STATE__ = 1

set_log_level(LogLevel.Debug)


def state_change_worker():
    while True:
        shared_state.__STATE__ += 1
        log(LogLevel.Info, "entrypoint", f'{shared_state.__STATE__}')
        time.sleep(1)


worker = threading.Thread(target=state_change_worker)
worker.start()

# os.environ["GST_DEBUG"] = "*:5"

import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib


def on_message(bus, message, loop):
    """Handles messages from the GStreamer bus."""
    t = message.type
    if t == Gst.MessageType.EOS:
        print("End-of-stream reached.")
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"Error: {err}, {debug}")
        loop.quit()


def main():
    arr = np.zeros((3, 3, 3), dtype=np.uint8)
    # print(arr)

    attr = Attribute(namespace="some", name="attr", hint="x", values=[
        AttributeValue.float(1.0, confidence=0.5),
    ])

    set_attributes([attr])

    # print(attr.json)

    # Initialize GStreamer
    Gst.init(None)

    # Create GStreamer pipeline
    pipeline = Gst.Pipeline.new("test-pipeline")

    if not pipeline:
        print("Failed to create pipeline.")
        return

    pipeline_str = ("videotestsrc pattern=black ! "
                    "video/x-raw, width=10, height=10 ! "
                    "rspy module=mymod class=MyHandler ! "
                    "fakesink sync=false")

    # Create the pipeline
    pipeline = Gst.parse_launch(pipeline_str)

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_message, loop)

    # Set the pipeline to the playing state
    pipeline.set_state(Gst.State.PLAYING)

    try:
        log(LogLevel.Info, "entrypoint", "Starting pipeline...")
        loop.run()
    except KeyboardInterrupt:
        log(LogLevel.Info, "entrypoint", "Stopping pipeline...")
    finally:
        pipeline.set_state(Gst.State.NULL)
