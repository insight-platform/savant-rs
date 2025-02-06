import gi

gi.require_version('Gst', '1.0')

from savant_rs.logging import log, LogLevel
from savant_rs.webserver.kvs import get_attribute
from savant_rs.gstreamer import GstBuffer
from savant_rs.primitives import Attribute, AttributeValue
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

counter = 0
last = time.time()


def run(buf: GstBuffer):
    global counter
    global last
    counter += 1
    if counter % 1000 == 0:
        log(LogLevel.Info, "mymod", f'Counter: {counter} Rate: {1000 / (time.time() - last)}')
        last = time.time()
    buf.replace_id_meta([1, 2, 3, 4])
    res = buf.id_meta
    assert res == [1, 2, 3, 4]

    thread_id = threading.get_ident()
    log(LogLevel.Debug, "mymod", f'Running on thread {thread_id}, Buffer w={buf.is_writable}')
