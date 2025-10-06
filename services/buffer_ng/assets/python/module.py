from typing import Any, Callable
from savant_rs.logging import log, LogLevel
from savant_rs.utils.serialization import Message
from time import sleep, time


class MessageHandler:
    """
    This handler is called for each message received from the ingress.
    """

    def __init__(self):
        self.count = 0
        self.now = time()

    def __call__(
        self, topic: str, message: Message
    ) -> (str, Message):
        """
        This handler is called for each message received from the ingress.

        :param message_id: unique message id across the service, allows to track the message between ingress and egress
        :param ingress_name: name of the ingress that received the message
        :param topic: ZMQ topic of the message if any
        :param message: message object, can be modified in place to add/remove labels, attributes, etc.
        :return: message object to be sent to egress
        """
        self.count += 1
        if self.count % 1000 == 0:
            print(f"message_handler {self.count}, elapsed {time() - self.now}s")
            self.now = time()
        return topic, message
        # return None


def init(params: Any) -> Callable:
    """
    This function is called once when the service starts. It is specified in the configuration.json file.
    """
    log(LogLevel.Info, "buffer_ng::init", "Buffer NG service initialized successfully")
    # True means that the service is initialized successfully and can start processing messages
    return MessageHandler()
    #return None
