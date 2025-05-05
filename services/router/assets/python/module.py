from savant_rs import register_handler, version
from savant_rs.logging import log, LogLevel

class Handler:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass

def init():
    log(LogLevel.Info, "router::init", f"Savant-Rs version: {version()}")
    register_handler("name",Handler())
    return True