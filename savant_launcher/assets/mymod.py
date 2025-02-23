import entrypoint

from savant_rs.logging import log, LogLevel
from savant_rs.gstreamer import InvocationReason, FlowResult

import time
import threading

NUM = 100000


class MyHandler:
    def __init__(self, element_name, **kwargs):
        self.counter = {}
        self.replacements = {}
        self.last = {}
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.element = entrypoint.pipeline.get_by_name(element_name)
        if self.element is None:
            print(f"Failed to find rspy element {element_name}")
            exit(1)

    def __call__(self, element_name: str, reason: InvocationReason):

        if reason == InvocationReason.Buffer:
            _ = self.element.get_property("current-buffer")
            thread_id = threading.get_ident()
            self.counter[thread_id] = self.counter.get(thread_id, 0) + 1
            if self.counter[thread_id] % NUM == 0:
                log(LogLevel.Info, "mymod",
                    f'Thread: {thread_id}, Element {element_name}, Counter: {self.counter.get(thread_id, 0)}, Replacements: {self.replacements.get(thread_id, 0)}, Rate: {NUM / (time.time() - self.last.get(thread_id, 0))}')
                self.last[thread_id] = time.time()
            return FlowResult.Ok

        if reason == InvocationReason.SourceEvent:
            _ = self.element.get_property("source-event")
            return True

        if reason == InvocationReason.SinkEvent:
            _ = self.element.get_property("sink-event")
            return True
