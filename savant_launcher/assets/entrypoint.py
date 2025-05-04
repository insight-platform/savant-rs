from savant_rs.logging import log, LogLevel, set_log_level
from savant_rs import register_handler
from savant_rs.gstreamer import InvocationReason, FlowResult
import time
import threading

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

set_log_level(LogLevel.Debug)

NUM = 100000


class PyFuncHandler:
    def __init__(self, pipeline, **kwargs):
        self.counter = {}
        self.replacements = {}
        self.last = {}
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.pipeline = pipeline
        self.element = None

    def __call__(self, element_name: str, reason: InvocationReason, *args):
        if self.element is None:
            self.element = self.pipeline.get_by_name(element_name)

        if reason == InvocationReason.Buffer:
            buf = self.element.get_property("current-buffer")
            thread_id = threading.get_ident()
            self.counter[thread_id] = self.counter.get(thread_id, 0) + 1
            if self.counter[thread_id] % NUM == 0:
                log(
                    LogLevel.Info,
                    "mymod",
                    f"Thread: {thread_id}, Element {element_name}, Counter: {self.counter.get(thread_id, 0)}, Replacements: {self.replacements.get(thread_id, 0)}, Rate: {NUM / (time.time() - self.last.get(thread_id, 0))}",
                )
                self.last[thread_id] = time.time()
            return FlowResult.Ok

        if reason == InvocationReason.SourceEvent:
            _ = self.element.get_property("source-event")
            return True

        if reason == InvocationReason.SinkEvent:
            _ = self.element.get_property("sink-event")
            return True

        if reason == InvocationReason.StateChange:
            current_state, next_state = args
            current_state = Gst.State(current_state).value_nick
            next_state = Gst.State(next_state).value_nick
            log(
                LogLevel.Info,
                "mymod",
                f"Element {element_name} state change: {current_state} -> {next_state}",
            )
            return True


def on_message(_, message, loop):
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
    # Create GStreamer pipeline (gst::init is called by the launcher)

    pipeline_str = """videotestsrc pattern=black ! 
                    video/x-raw, width=10, height=10 ! queue ! 
                    rspy name=rspy0 savant-pipeline-name=pipeline savant-pipeline-stage=rspy0 !  
                    rspy name=rspy1 savant-pipeline-name=pipeline savant-pipeline-stage=rspy1 !
                    fakesink sync=false"""

    # Create the pipeline
    pipeline = Gst.parse_launch(pipeline_str)

    rspy0_handler = PyFuncHandler(pipeline, some="parameter", other=1)
    register_handler("rspy0", rspy0_handler)

    rspy1_handler = PyFuncHandler(pipeline, some="parameter", other=1)
    register_handler("rspy1", rspy1_handler)

    return pipeline
