from typing import Any
from savant_rs import register_handler, version
from savant_rs.logging import log, LogLevel
from time import time
from savant_rs.primitives import VideoFrame
from typing import Optional
from savant_rs.utils.serialization import Message


class MergeHandler:
    def __call__(
        self,
        ingress: str,
        topic: str,
        video_frame: VideoFrame,
        labels: list[str],
        data: list[bytes],
        incoming_video_frame: Optional[VideoFrame],
        incoming_labels: Optional[list[str]],
        incoming_data: Optional[list[bytes]],
    ) -> bool:
        """
        This handler is called for each message received from the ingress.
        The current frame should be updated until the merging is considered as complete. When it is complete, the handler should return True.

        :param ingress: name of the ingress that received the message
        :param topic: ZMQ topic of the current frame if any
        :param video_frame: current video frame. This parameter must be updated with the incoming video frame.
        :param labels: labels currently assigned to the current frame. This parameter must be updated with the incoming labels.
        :param data: data currently assigned to the current frame. This parameter must be updated with the incoming data.
        :param incoming_video_frame: incoming video frame or None if the frame is the first one (automatically becomes the current frame and can be updated)
        :param incoming_labels: labels of the incoming frame or None if the frame is the first one
        :param incoming_data: data elements of the incoming frame or None if the frame is the first one
        :return: True if the merging is considered as complete and can be sent to the egress
        """
        return False


class HeadExpiredHandler:
    def __call__(self, frame: VideoFrame, labels: list[str], data: list[bytes]) -> Optional[Message]:
        """
        This handler is called when the head of the queue is expired.

        :param frame: video frame that is expired
        :param labels: labels of the frame
        :param data: data elements of the frame
        :return: True if send, False if drop
        """
        return True


class HeadReadyHandler:
    def __call__(self, frame: VideoFrame, labels: list[str], data: list[bytes]) -> Optional[Message]:
        """
        This handler is called when the head of the queue is ready.

        :param frame: video frame that is ready
        :param labels: labels of the frame
        :param data: data elements of the frame
        :return: True if send, False if drop
        """
        return True


class LateArrivalHandler:
    def __call__(self, frame: VideoFrame, labels: list[str], data: list[bytes]):
        """
        This handler is called when a new frame is late.

        :param frame: video frame that is late
        :param labels: labels of the frame
        :param data: data elements of the frame
        """
        pass


class UnsupportedMessageHandler:
    def __call__(self, ingress_name: str, topic: str, message: Message, data: list[bytes]):
        """
        This handler is called when a message is unsupported.

        :param ingress_name: name of the ingress that received the message
        :param topic: ZMQ topic of the message if any
        :param message: message object, can be modified in place to add/remove labels, attributes, etc.
        :param data: data elements of the message
        :return: message to be sent to the egress or None if the message should be dropped
        """
        log(LogLevel.Info, "meta_merge::unsupported_message_handler",
            f"Unsupported message received from {ingress_name} on topic {topic}, message: {message}, data: {data}")

class SendHandler:
    def __call__(self, topic: str, message: Message, data: list[bytes], labels: list[str]):
        """
        This handler is called when a message is ready to be sent to the egress.

        :param topic: topic of the message
        :param message: message object, can be modified in place to add/remove labels, attributes, etc.
        :param data: data elements of the message
        :param labels: labels of the message
        :return: True if send, False if drop
        """
        return True


def init(params: Any):
    """
    This function is called once when the service starts. It is specified in the configuration.json file.
    """
    log(LogLevel.Info, "meta_merge::init", "Initializing meta merge service")
    register_handler("merge_handler", MergeHandler())
    register_handler("head_expired_handler", HeadExpiredHandler())
    register_handler("head_ready_handler", HeadReadyHandler())
    register_handler("late_arrival_handler", LateArrivalHandler())
    register_handler("unsupported_message_handler",
                     UnsupportedMessageHandler())
    register_handler("send_handler", SendHandler())
    log(
        LogLevel.Info, "meta_merge::init", "Meta merge service initialized successfully"
    )
    # True means that the service is initialized successfully and can start processing messages
    return True
