from enum import Enum
from typing import List, Optional

from savant_rs.primitives import Attribute

__all__ = [
    "FlowResult",
    "InvocationReason",
]

class FlowResult(Enum):
    CustomSuccess2: ...
    CustomSuccess1: ...
    CustomSuccess: ...
    Ok: ...
    NotLinked: ...
    Flushing: ...
    Eos: ...
    NotNegotiated: ...
    Error: ...
    NotSupported: ...
    CustomError: ...
    CustomError1: ...
    CustomError2: ...

class InvocationReason(Enum):
    Buffer: ...
    SinkEvent: ...
    SourceEvent: ...
    StateChange: ...
    IngressMessageTransformer: ...
