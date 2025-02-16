import gi

gi.require_version('Gst', '1.0')

from savant_rs.logging import log, LogLevel
from savant_rs.webserver.kvs import get_attribute
from savant_rs.gstreamer import GstBuffer, FlowResult
import shared_state

import time
import threading

log(LogLevel.Info, "mymod", "Hello, world!")

attr = get_attribute("some", "attr")
print(attr)


def get_attr():
    while True:
        time.sleep(5)
        attr = get_attribute("some", "attr")
        log(LogLevel.Info, "mymod", attr.json)
        log(LogLevel.Info, "mymod", f'{shared_state.__STATE__}')


worker = threading.Thread(target=get_attr)
worker.start()

counter = {}
replacements = {}
last = {}


# def ih(element_name: str):
#     log(LogLevel.Info, "mymod", f'Init at {element_name}')
#     return True
#
#
# def bh(element_name: str, buf: GstBuffer):
#     global counter, last, replacements
#     thread_id = threading.get_ident()
#     counter[thread_id] = counter.get(thread_id, 0) + 1
#     if counter[thread_id] % 1000 == 0:
#         log(LogLevel.Info, "mymod",
#             f'Thread: {thread_id}, Element {element_name}, Counter: {counter.get(thread_id, 0)}, Replacements: {replacements.get(thread_id, 0)}, Rate: {1000 / (time.time() - last.get(thread_id, 0))}')
#         last[thread_id] = time.time()
#     prev_meta = buf.id_meta
#     if prev_meta:
#         replacements[thread_id] = replacements.get(thread_id, 0) + 1
#         buf.replace_id_meta([1, 2, 3, 4])
#         res = buf.id_meta
#         assert res == [1, 2, 3, 4]
#     else:
#         buf.replace_id_meta([0, 1, 2, 3])
#
#     return FlowResult.Ok
#
#
# def eh(element_name: str):
#     log(LogLevel.Info, "mymod", f'Event at {element_name}: not-yet-implemented')
#     return True


class MyHandler:
    def __init__(self, **kwargs):
        self.counter = {}
        self.replacements = {}
        self.last = {}
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __call__(self, element_name: str, buf: GstBuffer):
        thread_id = threading.get_ident()
        self.counter[thread_id] = self.counter.get(thread_id, 0) + 1
        if self.counter[thread_id] % 1000 == 0:
            log(LogLevel.Info, "mymod",
                f'Thread: {thread_id}, Element {element_name}, Counter: {self.counter.get(thread_id, 0)}, Replacements: {self.replacements.get(thread_id, 0)}, Rate: {1000 / (time.time() - self.last.get(thread_id, 0))}')
            self.last[thread_id] = time.time()
        prev_meta = buf.id_meta
        if prev_meta:
            self.replacements[thread_id] = self.replacements.get(thread_id, 0) + 1
            buf.replace_id_meta([1, 2, 3, 4])
            res = buf.id_meta
            assert res == [1, 2, 3, 4]
        else:
            buf.replace_id_meta([0, 1, 2, 3])
        mem = buf.memory(0).read_with(lambda x: x)
        return FlowResult.Ok
