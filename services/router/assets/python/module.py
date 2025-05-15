from typing import Any
from savant_rs import register_handler, version
from savant_rs.logging import log, LogLevel


class Handler:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass


def init(params: Any):
    print(f"params: {params}")
    log(LogLevel.Info, "router::init", f"savant-rs version: {version()}")
    register_handler("name", Handler())
    return True
