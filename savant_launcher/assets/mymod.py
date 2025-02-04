from savant_rs.logging import log, LogLevel
from savant_rs.webserver.kvs import get_attribute
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


def run():
    thread_id = threading.get_ident()
    # raise Exception("This is an exception")
    log(LogLevel.Info, "mymod", f'Running on thread {thread_id}')
