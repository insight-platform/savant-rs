from enum import Enum
from typing import Optional, List

from .attribute_value import AttributeValue, AttributeValueView

__all__ = [
    'Attribute',
    'AttributeUpdatePolicy',
]

class Attribute:
    values: List[AttributeValue]

    def __init__(self,
                 namespace: str,
                 name: str,
                 values: List[AttributeValue],
                 hint: Optional[str],
                 is_persistent: bool = True,
                 is_hidden: bool = False): ...

    @classmethod
    def persistent(cls,
                   namespace: str,
                   name: str,
                   values: List[AttributeValue],
                   hint: Optional[str] = None,
                   is_hidden: bool = False): ...

    @classmethod
    def temporary(cls,
                  namespace: str,
                  name: str,
                  values: List[AttributeValue],
                  hint: Optional[str] = None,
                  is_hidden: bool = False): ...

    def is_temporary(self) -> bool: ...

    def is_hidden(self) -> bool: ...

    def make_peristent(self): ...

    def make_temporary(self): ...

    @property
    def namespace(self) -> str: ...

    @property
    def name(self) -> str: ...

    @property
    def values_view(self) -> AttributeValueView: ...

    @property
    def hint(self) -> Optional[str]: ...

    @property
    def json(self) -> str: ...

    @classmethod
    def from_json(cls, json: str) -> 'Attribute': ...


class AttributeUpdatePolicy(Enum):
    ReplaceWithForeignWhenDuplicate: ...
    KeepOwnWhenDuplicate: ...
    ErrorWhenDuplicate: ... 