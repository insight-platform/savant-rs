import gi

gi.require_version('Gst', '1.0')

from savant_rs.logging import log, LogLevel
from savant_rs.webserver.kvs import get_attribute
from savant_rs.gstreamer import GstBuffer
import shared_state

import time
import threading

# from pyvips.gobject import GObject

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


def run(element_name: str, buf: GstBuffer):
    global counter, last, replacements
    thread_id = threading.get_ident()
    counter[thread_id] = counter.get(thread_id, 0) + 1
    if counter[thread_id] % 1000 == 0:
        log(LogLevel.Info, "mymod",
            f'Thread: {thread_id}, Element {element_name}, Counter: {counter.get(thread_id, 0)}, Replacements: {replacements.get(thread_id, 0)}, Rate: {1000 / (time.time() - last.get(thread_id, 0))}')
        last[thread_id] = time.time()
    prev_meta = buf.id_meta
    if prev_meta:
        replacements[thread_id] = replacements.get(thread_id, 0) + 1
        buf.replace_id_meta([1, 2, 3, 4])
        res = buf.id_meta
        assert res == [1, 2, 3, 4]
    else:
        buf.replace_id_meta([0, 1, 2, 3])
