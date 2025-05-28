from typing import Any
from savant_rs import register_handler, version
from savant_rs.logging import log, LogLevel
from savant_rs.utils.serialization import Message
from time import time


class IngressHandler:
    def __init__(self):
        self.count = 0
        self.now = time()
        pass

    def __call__(self, *args, **kwargs):
        message: Message = args[2]
        ingress_name: str = args[0]
        message.labels = ["label1", "label2"]
        self.count += 1
        if self.count % 1000 == 0:
            print(f"ingress_handler {self.count}, elapse {time() - self.now}s")
            self.now = time()
        return message


class EgressSourceHandler:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        print("egress_source_handler", args, kwargs)
        return args[1]


class EgressTopicHandler:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        print("egress_topic_handler", args, kwargs)
        return args[1]


def init(params: Any):
    log(LogLevel.Info, "router::init", f"savant-rs version: {version()}")
    register_handler("ingress_handler", IngressHandler())
    register_handler("egress_source_handler", EgressSourceHandler())
    register_handler("egress_topic_handler", EgressTopicHandler())
    return True
