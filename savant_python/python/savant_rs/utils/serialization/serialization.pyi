from typing import Optional, Union

from savant_rs.primitives import (EndOfStream, Shutdown, UserData, VideoFrame,
                                  VideoFrameBatch, VideoFrameUpdate)
from savant_rs.utils import ByteBuffer

__all__ = [
    "Message",
    "save_message",
    "save_message_to_bytebuffer",
    "save_message_to_bytes",
    "load_message",
    "load_message_from_bytebuffer",
    "load_message_from_bytes",
    "clear_source_seq_id",
]

class Message:
    """A class representing different types of messages in the Savant system."""

    @staticmethod
    def unknown(s: str) -> "Message":
        """Create a new undefined message.

        Parameters
        ----------
        s : str
            The message text

        Returns
        -------
        Message
            The message of Unknown type
        """
        ...

    @staticmethod
    def shutdown(shutdown: Shutdown) -> "Message":
        """Create a new shutdown message.

        Parameters
        ----------
        shutdown : Shutdown
            The shutdown message

        Returns
        -------
        Message
            The message of Shutdown type
        """
        ...

    @staticmethod
    def user_data(t: UserData) -> "Message":
        """Create a new user data message.

        Parameters
        ----------
        t : UserData
            The user data message

        Returns
        -------
        Message
            The message of UserData type
        """
        ...

    @staticmethod
    def video_frame(frame: VideoFrame) -> "Message":
        """Create a new video frame message.

        Parameters
        ----------
        frame : VideoFrame
            The video frame

        Returns
        -------
        Message
            The message of VideoFrame type
        """
        ...

    @staticmethod
    def video_frame_update(update: VideoFrameUpdate) -> "Message":
        """Create a new video frame update message.

        Parameters
        ----------
        update : VideoFrameUpdate
            The update struct

        Returns
        -------
        Message
            The message of VideoFrameUpdate type
        """
        ...

    @staticmethod
    def video_frame_batch(batch: VideoFrameBatch) -> "Message":
        """Create a new video frame batch message.

        Parameters
        ----------
        batch : VideoFrameBatch
            The video frame batch

        Returns
        -------
        Message
            The message of VideoFrameBatch type
        """
        ...

    @staticmethod
    def end_of_stream(eos: EndOfStream) -> "Message":
        """Create a new end of stream message.

        Parameters
        ----------
        eos : EndOfStream
            The end of stream message

        Returns
        -------
        Message
            The message of EndOfStream type
        """
        ...

    def is_unknown(self) -> bool:
        """Checks if the message is of Unknown type.

        Returns
        -------
        bool
            True if the message is of Unknown type, False otherwise
        """
        ...

    def is_shutdown(self) -> bool:
        """Checks if the message is of Shutdown type.

        Returns
        -------
        bool
            True if the message is of Shutdown type, False otherwise
        """
        ...

    def is_user_data(self) -> bool:
        """Checks if the message is of UserData type.

        Returns
        -------
        bool
            True if the message is of UserData type, False otherwise
        """
        ...

    def is_video_frame(self) -> bool:
        """Checks if the message is of VideoFrame type.

        Returns
        -------
        bool
            True if the message is of VideoFrame type, False otherwise
        """
        ...

    def is_video_frame_update(self) -> bool:
        """Checks if the message is of VideoFrameUpdate type.

        Returns
        -------
        bool
            True if the message is of VideoFrameUpdate type, False otherwise
        """
        ...

    def is_video_frame_batch(self) -> bool:
        """Checks if the message is of VideoFrameBatch type.

        Returns
        -------
        bool
            True if the message is of VideoFrameBatch type, False otherwise
        """
        ...

    def is_end_of_stream(self) -> bool:
        """Checks if the message is of EndOfStream type.

        Returns
        -------
        bool
            True if the message is of EndOfStream type, False otherwise
        """
        ...

    def as_user_data(self) -> Optional[UserData]:
        """Returns the message as UserData type.

        Returns
        -------
        Optional[UserData]
            The message as UserData type if it is of that type, None otherwise
        """
        ...

    def as_video_frame(self) -> Optional[VideoFrame]:
        """Returns the message as VideoFrame type.

        Returns
        -------
        Optional[VideoFrame]
            The message as VideoFrame type if it is of that type, None otherwise
        """
        ...

    def as_video_frame_update(self) -> Optional[VideoFrameUpdate]:
        """Returns the message as VideoFrameUpdate type.

        Returns
        -------
        Optional[VideoFrameUpdate]
            The message as VideoFrameUpdate type if it is of that type, None otherwise
        """
        ...

    def as_video_frame_batch(self) -> Optional[VideoFrameBatch]:
        """Returns the message as VideoFrameBatch type.

        Returns
        -------
        Optional[VideoFrameBatch]
            The message as VideoFrameBatch type if it is of that type, None otherwise
        """
        ...

def save_message(message: Message, no_gil: bool = True) -> bytes:
    """Save a message to a byte array.

    Parameters
    ----------
    message : Message
        The message to save
    no_gil : bool, optional
        Whether to release the GIL while saving the message, by default True

    Returns
    -------
    bytes
        The byte array containing the message
    """
    ...

def save_message_to_bytebuffer(
    message: Message, with_hash: bool = True, no_gil: bool = True
) -> ByteBuffer:
    """Save a message to a byte buffer.

    Parameters
    ----------
    message : Message
        The message to save
    with_hash : bool, optional
        Whether to include a hash of the message in the returned byte buffer, by default True
    no_gil : bool, optional
        Whether to release the GIL while saving the message, by default True

    Returns
    -------
    ByteBuffer
        The byte buffer containing the message
    """
    ...

def save_message_to_bytes(message: Message, no_gil: bool = True) -> bytes:
    """Save a message to python bytes.

    Parameters
    ----------
    message : Message
        The message to save
    no_gil : bool, optional
        Whether to release the GIL while saving the message, by default True

    Returns
    -------
    bytes
        The byte buffer containing the message
    """
    ...

def load_message(bytes: Union[bytes, bytearray], no_gil: bool = True) -> Message:
    """Loads a message from a byte array.

    Parameters
    ----------
    bytes : Union[bytes, bytearray]
        The byte array to load the message from
    no_gil : bool, optional
        Whether to release the GIL while loading the message, by default True

    Returns
    -------
    Message
        The loaded message
    """
    ...

def load_message_from_bytebuffer(buffer: ByteBuffer, no_gil: bool = True) -> Message:
    """Loads a message from a ByteBuffer.

    Parameters
    ----------
    buffer : ByteBuffer
        The byte buffer to load the message from
    no_gil : bool, optional
        Whether to release the GIL while loading the message, by default True

    Returns
    -------
    Message
        The loaded message
    """
    ...

def load_message_from_bytes(buffer: bytes, no_gil: bool = True) -> Message:
    """Loads a message from python bytes.

    Parameters
    ----------
    buffer : bytes
        The byte buffer to load the message from
    no_gil : bool, optional
        Whether to release the GIL while loading the message, by default True

    Returns
    -------
    Message
        The loaded message
    """
    ...

def clear_source_seq_id() -> None:
    """Clears the source sequence ID.

    This function allows validating the sequence ID of messages.
    """
    ...
