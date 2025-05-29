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

    def __call__(self, message_id: int, ingress_name: str, topic: str, message: Message):
        if topic == "test1":
            message.labels = ["label1", "label2"]

        elif topic == "test2":
            message.labels = ["label1", "label3"]

        self.count += 1
        if self.count % 1000 == 0:
            print(f"ingress_handler {self.count}, elapsed {time() - self.now}s")
            self.now = time()
        return message


class EgressSourceHandler:
    def __init__(self):
        pass

    def __call__(self, message_id: int, egress_name: str, source: str, labels: list[str]):
        print(f"egress_source_handler message_id: {message_id}, egress_name: {egress_name}, source: {source}, labels: {labels}")
        return source


class EgressTopicHandler:
    def __init__(self):
        pass

    def __call__(self, message_id: int, egress_name: str, topic: str, labels: list[str]):
        print(f"egress_topic_handler message_id: {message_id}, egress_name: {egress_name}, topic: {topic}, labels: {labels}")
        return topic


def init(params: Any):
    log(LogLevel.Info, "router::init", f"savant-rs version: {version()}")
    register_handler("ingress_handler", IngressHandler())
    register_handler("egress_source_handler", EgressSourceHandler())
    register_handler("egress_topic_handler", EgressTopicHandler())
    return True
