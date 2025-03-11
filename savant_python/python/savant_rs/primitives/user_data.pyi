from typing import Optional, List, Tuple
from savant_rs.utils.serialization import Message
from .attribute import Attribute
from .attribute_value import AttributeValue

__all__ = [
    'UserData',
]

class UserData:
    def __init__(self, source_id: str): ...

    @property
    def source_id(self) -> str: ...

    @property
    def json(self) -> str: ...

    def to_message(self) -> Message: ...

    @property
    def attributes(self) -> List[Tuple[str, str]]: ...

    def get_attribute(self,
                      namespace: str,
                      name: str) -> Optional[Attribute]: ...

    def find_attributes_with_ns(self,
                                namespace: str) -> List[Tuple[str, str]]: ...

    def find_attributes_with_names(self,
                                   names: List[str]) -> List[Tuple[str, str]]: ...

    def find_attributes_with_hints(self,
                                   hints: List[Optional[str]]) -> List[Tuple[str, str]]: ...

    def delete_attributes_with_ns(self, namespace: str): ...

    def delete_attributes_with_names(self, names: List[str]): ...

    def delete_attributes_with_hints(self,
                                     hints: List[Optional[str]]): ...

    def delete_attribute(self, namespace: str, name: str) -> Optional[Attribute]: ...

    def set_attribute(self, attribute: Attribute) -> Optional[Attribute]: ...

    def set_persistent_attribute(self,
                                 namespace: str,
                                 name: str,
                                 is_hidden: bool,
                                 hint: Optional[str],
                                 values: Optional[List[AttributeValue]]): ...

    def set_temporary_attribute(self,
                                namespace: str,
                                name: str,
                                is_hidden: bool,
                                hint: Optional[str],
                                values: Optional[List[AttributeValue]]): ...

    def clear_attributes(self): ...

    @property
    def json_pretty(self) -> str: ...

    def to_protobuf(self, no_gil: bool = True) -> bytes: ...

    @classmethod
    def from_protobuf(cls,
                      protobuf: bytes,
                      no_gil: bool = True) -> 'UserData': ... 