from savant_rs.logging import log, LogLevel
from savant_rs.primitives import AttributeValue, Attribute
import numpy as np
import cv2

import os

# os.environ["GST_PLUGIN_PATH"] = "/home/ivan/dev/savant/savant-rs/dist/build_artifacts"
# os.environ["GST_DEBUG"] = "rsidentity:7"

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
        AttributeValue.bytes(dims=[8, 3, 8, 8], blob=bytes(3 * 8 * 8), confidence=None),
        AttributeValue.bytes_from_list(dims=[4, 1], blob=[0, 1, 2, 3], confidence=None),
        AttributeValue.integer(1, confidence=0.5),
        AttributeValue.float(1.0, confidence=0.5),
    ])

    # print(attr.json)

    # Initialize GStreamer
    Gst.init(None)

    # Create GStreamer pipeline
    pipeline = Gst.Pipeline.new("test-pipeline")

    if not pipeline:
        print("Failed to create pipeline.")
        return

    # Create elements manually
    source = Gst.ElementFactory.make("videotestsrc", "source")
    identity = Gst.ElementFactory.make("rsidentity", "identity-filter")
    capsfilter = Gst.ElementFactory.make("capsfilter", "filter")
    sink = Gst.ElementFactory.make("autovideosink", "sink")

    if not source or not identity or not capsfilter or not sink:
        print("Failed to create elements.")
        return

    # Set properties
    source.set_property("pattern", 0)  # 0 = Default test pattern

    # Configure capsfilter (optional)
    caps = Gst.Caps.from_string("video/x-raw, width=640, height=480, framerate=30/1")
    capsfilter.set_property("caps", caps)

    # Add elements to pipeline
    pipeline.add(source)
    pipeline.add(identity)
    pipeline.add(capsfilter)
    pipeline.add(sink)

    # Manually link elements
    if not source.link(identity):
        print("Failed to link source to identity.")
        return
    if not identity.link(capsfilter):
        print("Failed to link identity to capsfilter.")
        return
    if not capsfilter.link(sink):
        print("Failed to link capsfilter to sink.")
        return

    # Start the pipeline
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_message, loop)

    pipeline.set_state(Gst.State.PLAYING)

    try:
        log(LogLevel.Info, "entrypoint", "Starting pipeline...")
        loop.run()
    except KeyboardInterrupt:
        log(LogLevel.Info, "entrypoint", "Stopping pipeline...")
    finally:
        pipeline.set_state(Gst.State.NULL)


main()
