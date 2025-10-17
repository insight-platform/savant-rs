from typing import Any, Dict, List, Union
from savant_rs import register_handler
from savant_rs.logging import log, LogLevel
from savant_rs.primitives import VideoFrame, EndOfStream
from savant_rs.services.ffmpeg_source import StreamInfoVideoFile, StreamInfoRTSP, StreamProperties


class FrameEventHandler:
    """
    This handler is called for each frame event.
    """
    
    def __call__(self, video_frame: VideoFrame) -> bool:
        # video_frame can be modified here
        return True # if frame must be sent


class ProbeEventHandler:
    """
    This handler is called for each probe event.
    """
    
    def __call__(self, source_id: str, si: StreamProperties) -> bool:
        return True # if permitted to proceed with the stream


class StreamTerminationEventHandler:
    """
    This handler is called for each stream termination event.
    """
    
    def __call__(self, source_id: str, si: StreamProperties) -> Union[None, EndOfStream]:
        return None # if permitted to proceed with the stream

class CreateStreamsRequestEventHandler:
    """
    This handler is called for each create streams request event.
    """
    def __call__(self) -> List[Union[StreamInfoVideoFile, StreamInfoRTSP]]:
        return [] # list of streams to create

class StopStreamsRequestEventHandler:
    """
    This handler is called for each stop streams request event.
    """
    def __call__(self) -> List[str]:
        return [] # list of sources to terminate

def init(params: Any):
    """
    This function is called once when the service starts. It is specified in the configuration.json file.
    """
    log(LogLevel.Info, "ffmpeg_source::init::python", "Initializing FFmpeg source service")
    register_handler("frame_event", FrameEventHandler())
    register_handler("probe_event", ProbeEventHandler())
    register_handler("stream_termination_event", StreamTerminationEventHandler())
    register_handler("create_streams_request", CreateStreamsRequestEventHandler())
    register_handler("stop_streams_request", StopStreamsRequestEventHandler())
    log(LogLevel.Info, "ffmpeg_source::init::python", "FFmpeg source service initialized successfully")
    # True means that the service is initialized successfully and can start processing messages
    return True
