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
        ingress_name: str, 
        topic: str, 
        current: VideoFrame, 
        current_tags: list[str], 
        incoming: Optional[VideoFrame], 
        incoming_tags: Optional[list[str]]
    ) -> bool:
        """
        This handler is called for each message received from the ingress. 
        The current frame should be updated until the merging is considered as complete. When it is complete, the handler should return True.

        :param ingress_name: name of the ingress that received the message
        :param topic: ZMQ topic of the message if any
        :param current: current video frame
        :param current_tags: tags currently assigned to the current frame
        :param incoming: incoming video frame or None if the frame is the first one (automatically becomes the current frame and can be updated)
        :param incoming_tags: tags of the incoming frame or None if the frame is the first one
        :return: True if the merging is complete and can be sent to the egress
        """
        return False


class HeadExpiredHandler:
    def __call__(
        self, frame: VideoFrame, tags: list[str]
    ) -> Optional[Message]:
        """
        This handler is called when the head of the queue is expired.

        :param frame: video frame that is expired
        :param tags: tags of the frame
        :return: message to be sent to the egress or None if the frame should be dropped
        """
        return frame.to_message()

class HeadReadyHandler:
    def __call__(
        self, frame: VideoFrame, tags: list[str]
    ) -> Optional[Message]:
        """
        This handler is called when the head of the queue is ready.

        :param frame: video frame that is ready
        :param tags: tags of the frame
        :return: message to be sent to the egress or None if the frame should be dropped
        """
        return frame.to_message()


class LateArrivalHandler:
    def __call__(
        self, frame: VideoFrame, tags: list[str]
    ):
        """
        This handler is called when a new frame is late.

        :param frame: video frame that is late
        :param tags: tags of the frame
        """
        pass


def init(params: Any):
    """
    This function is called once when the service starts. It is specified in the configuration.json file.
    """
    log(LogLevel.Info, "meta_merge::init", "Initializing meta merge service")
    register_handler("merge_handler", MergeHandler())
    register_handler("head_expired_handler", HeadExpiredHandler())
    register_handler("head_ready_handler", HeadReadyHandler())
    register_handler("late_arrival_handler", LateArrivalHandler())
    log(LogLevel.Info, "meta_merge::init", "Meta merge service initialized successfully")
    # True means that the service is initialized successfully and can start processing messages
    return True
