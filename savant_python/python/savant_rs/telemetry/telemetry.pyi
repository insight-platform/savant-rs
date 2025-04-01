from enum import Enum
from typing import Optional

__all__ = [
    "ContextPropagationFormat",
    "Protocol",
    "Identity",
    "ClientTlsConfig",
    "TracerConfiguration",
    "TelemetryConfiguration",
    "init",
    "shutdown",
    "init_from_file",
]

class ContextPropagationFormat(Enum):
    Jaeger: int
    W3C: int

class Protocol(Enum):
    Grpc: int
    HttpBinary: int
    HttpJson: int

class Identity:
    def __init__(self, key: str, certificate: str): ...

class ClientTlsConfig:
    def __init__(
        self, certificate: Optional[str] = None, identity: Optional[Identity] = None
    ): ...

class TracerConfiguration:
    def __init__(
        self,
        service_name: str,
        protocol: Protocol,
        endpoint: str,
        tls: Optional[ClientTlsConfig] = None,
        timeout: Optional[int] = None,
    ): ...

class TelemetryConfiguration:
    def __init__(
        self,
        context_propagation_format: Optional[ContextPropagationFormat] = None,
        tracer: Optional[TracerConfiguration] = None,
    ): ...
    @classmethod
    def no_op(cls) -> TelemetryConfiguration: ...

def init(config: TelemetryConfiguration) -> None: ...
def init_from_file(path: str) -> None: ...
def shutdown() -> None: ...
