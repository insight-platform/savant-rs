from typing import Any
from savant_rs import register_handler, version
from savant_rs.logging import log, LogLevel
from savant_rs.utils.serialization import Message
from time import time


class IngressHandler:
    """
    This handler is called for each message received from the ingress.
    """

    def __init__(self):
        self.count = 0
        self.now = time()
        pass

    def __call__(
        self, message_id: int, ingress_name: str, topic: str, message: Message
    ):
        """
        This handler is called for each message received from the ingress.

        :param message_id: unique message id across the service, allows to track the message between ingress and egress
        :param ingress_name: name of the ingress that received the message
        :param topic: ZMQ topic of the message if any
        :param message: message object, can be modified in place to add/remove labels, attributes, etc.
        :return: message object to be sent to egress
        """
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
    """
    This handler is called periodically for egress to modify the source of the message if needed. It is only called for video frames when keyframes,
    when the last value is expired. Thus, it is not called for every message.
    """

    def __init__(self):
        pass

    def __call__(
        self, message_id: int, egress_name: str, source: str, labels: list[str]
    ):
        """
        This handler is called periodically for egress to modify the source of the message if needed. It is only called for video frames when keyframes,
        when the last value is expired. Thus, it is not called for every message.

        :param message_id: unique message id across the service, allows to track the message between ingress and egress handlers
        :param egress_name: name of the egress that processes the message
        :param source: source of the message
        :param labels: labels of the message
        :return: source of the message to be sent to the egress
        """
        print(
            f"egress_source_handler message_id: {message_id}, egress_name: {egress_name}, source: {source}, labels: {labels}"
        )
        return source


class EgressTopicHandler:
    """
    This handler is called periodically for egress to modify the topic of the message if needed. It is only called for video frames when keyframes,
    when the last value is expired. Thus, it is not called for every message.
    """

    def __init__(self):
        pass

    def __call__(
        self, message_id: int, egress_name: str, topic: str, labels: list[str]
    ):
        """
        This handler is called periodically for egress to modify the topic of the message if needed. It is only called for video frames when keyframes,
        when the last value is expired. Thus, it is not called for every message.

        :param message_id: unique message id across the service, allows to track the message between ingress and egress handlers
        :param egress_name: name of the egress that processes the message
        :param topic: topic of the message
        :param labels: labels of the message
        :return: topic of the message to be sent to the next egress
        """
        print(
            f"egress_topic_handler message_id: {message_id}, egress_name: {egress_name}, topic: {topic}, labels: {labels}"
        )
        return topic


def init(params: Any):
    """
    This function is called once when the service starts. It is specified in the configuration.json file.
    """
    log(LogLevel.Info, "router::init", "Initializing router service")
    register_handler("ingress_handler", IngressHandler())
    register_handler("egress_source_handler", EgressSourceHandler())
    register_handler("egress_topic_handler", EgressTopicHandler())
    log(LogLevel.Info, "router::init", "Router service initialized successfully")
    # True means that the service is initialized successfully and can start processing messages
    return True
