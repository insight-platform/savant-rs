from savant_rs.utils.serialization import Message

__all__ = [
    'EndOfStream',
]

class EndOfStream:
    def __init__(self, source_id: str): ...

    @property
    def source_id(self) -> str: ...

    @property
    def json(self) -> str: ...

    def to_message(self) -> Message: ... 