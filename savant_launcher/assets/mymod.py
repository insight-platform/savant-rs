import ctypes

import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

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

ref = None


def run(buf: GstBuffer):
    global ref
    ref = buf.copy()
    thread_id = threading.get_ident()
    log(LogLevel.Info, "mymod", f'Running on thread {thread_id}, Buffer {buf.dts_or_pts}')
