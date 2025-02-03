import os

# import savant_plugin_sample

os.environ["GST_PLUGIN_PATH"] = "/mnt/development/build/debug"
os.environ["GST_DEBUG"] = "rsidentity:7"

import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib


# GST_PLUGIN_PATH=/mnt/development/build/debug


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
        print("Running GStreamer pipeline with identity element...")
        loop.run()
    except KeyboardInterrupt:
        print("Stopping pipeline...")
    finally:
        pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    main()
