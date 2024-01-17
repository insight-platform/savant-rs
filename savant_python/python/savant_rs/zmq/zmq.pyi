from enum import Enum
from typing import Optional, Union

from savant_rs.utils.serialization import Message


class WriterSocketType(Enum):
    Pub: int
    Dealer: int
    Req: int


class ReaderSocketType(Enum):
    Sub: int
    Router: int
    Rep: int


class TopicPrefixSpec:
    @staticmethod
    def source_id(topic: str) -> TopicPrefixSpec: ...

    @staticmethod
    def prefix(prefix: str) -> TopicPrefixSpec: ...

    @staticmethod
    def none() -> TopicPrefixSpec: ...


class WriterConfig:
    @property
    def endpoint(self) -> str: ...

    @property
    def socket_type(self) -> WriterSocketType: ...

    @property
    def bind(self) -> bool: ...

    @property
    def send_timeout(self) -> int: ...

    @property
    def receive_timeout(self) -> int: ...

    @property
    def receive_retries(self) -> int: ...

    @property
    def send_retries(self) -> int: ...

    @property
    def send_hwm(self) -> int: ...

    @property
    def receive_hwm(self) -> int: ...

    @property
    def fix_ipc_permissions(self) -> Optional[bool]: ...


class WriterConfigBuilder:
    def __init__(self, url: str): ...

    def with_socket_type(self, socket_type: WriterSocketType): ...

    def with_bind(self, bind: bool): ...

    def with_send_timeout(self, send_timeout: int): ...

    def with_receive_timeout(self, receive_timeout: int): ...

    def with_receive_retries(self, receive_retries: int): ...

    def with_send_retries(self, send_retries: int): ...

    def with_send_hwm(self, send_hwm: int): ...

    def with_receive_hwm(self, receive_hwm: int): ...

    def with_fix_ipc_permissions(self, fix_ipc_permissions: Optional[bool]): ...

    def build(self) -> WriterConfig: ...


class ReaderConfig:
    @property
    def endpoint(self) -> str: ...

    @property
    def socket_type(self) -> ReaderSocketType: ...

    @property
    def bind(self) -> bool: ...

    @property
    def receive_timeout(self) -> int: ...

    @property
    def receive_hwm(self) -> int: ...

    @property
    def topic_prefix_spec(self) -> TopicPrefixSpec: ...

    @property
    def routing_cache_size(self) -> int: ...

    @property
    def fix_ipc_permissions(self) -> Optional[bool]: ...


class ReaderConfigBuilder:
    def __init__(self, url: str): ...

    def with_socket_type(self, socket_type: ReaderSocketType): ...

    def with_bind(self, bind: bool): ...

    def with_receive_timeout(self, receive_timeout: int): ...

    def with_receive_hwm(self, receive_hwm: int): ...

    def with_topic_prefix_spec(self, topic_prefix: TopicPrefixSpec): ...

    def with_routing_cache_size(self, routing_cache_size: int): ...

    def with_fix_ipc_permissions(self, fix_ipc_permissions: Optional[bool]): ...

    def build(self) -> ReaderConfig: ...


class WriterResultSendTimeout:
    pass


class WriterResultActTimeout:
    timeout: int


class WriterResultAck:
    send_retries_spent: int
    receive_retries_spent: int
    time_spent: int


class WriterResultSuccess:
    retries_spent: int
    time_spent: int


class ReaderResultMessage:
    message: Message
    topic: bytes
    routing_id: Optional[bytes]

    def data_len(self) -> int: ...

    def data(self, index: int) -> bytes: ...


class ReaderResultEndOfStream:
    topic: bytes
    routing_id: Optional[bytes]


class ReaderResultTimeout:
    pass


class ReaderResultPrefixMismatch:
    topic: bytes
    routing_id: Optional[bytes]


class BlockingWriter:
    def __init__(self, config: WriterConfig): ...

    def is_started(self) -> bool: ...

    def start(self) -> None: ...

    def shutdown(self) -> None: ...

    def send_eos(self, topic: str) -> None: ...

    def send_message(self, topic: str, message: Message) -> Union[
        WriterResultSendTimeout, WriterResultActTimeout, WriterResultAck, WriterResultSuccess]: ...


class BlockingReader:
    def __init__(self, config: ReaderConfig): ...

    def is_started(self) -> bool: ...

    def start(self) -> None: ...

    def shutdown(self) -> None: ...

    def receive(self) -> Union[
        ReaderResultMessage, ReaderResultEndOfStream, ReaderResultTimeout, ReaderResultPrefixMismatch]: ...


class WriteOperationResult:
    def get(self) -> Union[WriterResultSendTimeout, WriterResultActTimeout, WriterResultAck, WriterResultSuccess]: ...

    def try_get(self) -> Optional[
        Union[WriterResultSendTimeout, WriterResultActTimeout, WriterResultAck, WriterResultSuccess]]: ...


class NonBlockingWriter:
    def __init__(self, config: WriterConfig, max_inflight_messages: int): ...

    def is_started(self) -> bool: ...

    def is_shutdown(self) -> bool: ...

    def start(self) -> None: ...

    def shutdown(self) -> None: ...

    def send_eos(self, topic: str) -> WriteOperationResult: ...

    def send_message(self, topic: str, message: Message) -> WriteOperationResult: ...

    def inflight_messages(self) -> int: ...


class NonBlockingReader:
    def __init__(self, config: ReaderConfig, results_queue_size: int): ...

    def is_started(self) -> bool: ...

    def is_shutdown(self) -> bool: ...

    def start(self) -> None: ...

    def shutdown(self) -> None: ...

    def receive(self) -> Union[
        ReaderResultMessage, ReaderResultEndOfStream, ReaderResultTimeout, ReaderResultPrefixMismatch]: ...

    def try_receive(self) -> Optional[
        Union[ReaderResultMessage, ReaderResultEndOfStream, ReaderResultTimeout, ReaderResultPrefixMismatch]]: ...

    def enqueued_results(self) -> int: ...
