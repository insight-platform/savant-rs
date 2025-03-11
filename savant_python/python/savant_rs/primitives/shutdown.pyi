from savant_rs.utils.serialization import Message

__all__ = [
    'Shutdown',
]

class Shutdown:
    def __init__(self, auth: str): ...

    @property
    def auth(self) -> str: ...

    @property
    def json(self) -> str: ...

    def to_message(self) -> Message: ... 