from typing import List, Optional, Union

from savant_rs.primitives import Attribute

__all__ = [
    'set_attributes',
    'search_attributes',
    'search_keys',
    'del_attributes',
    'get_attribute',
    'del_attribute',
    'serialize_attributes',
    'deserialize_attributes',
    'KvsSetOperation',
    'KvsDeleteOperation',
    'KvsSubscription',
]

def set_attributes(attributes: List[Attribute], ttl: Optional[int]) -> None: ...
def search_attributes(
    ns: Optional[str], name: Optional[str], no_gil: bool
) -> List[Attribute]: ...

# pub fn search_keys(ns: &Option<String>, name: &Option<String>) -> Vec<(String, String)>
def search_keys(ns: Optional[str], name: Optional[str], no_gil: bool) -> List[str]: ...

# pub fn del_attributes(ns: &Option<String>, name: &Option<String>)
def del_attributes(ns: Optional[str], name: Optional[str], no_gil: bool) -> None: ...

# pub fn get_attribute(ns: &str, name: &str) -> Option<Attribute>
def get_attribute(ns: str, name: str) -> Optional[Attribute]: ...

# pub fn del_attribute(ns: &str, name: &str) -> Option<Attribute>
def del_attribute(ns: str, name: str) -> Optional[Attribute]: ...

# pub fn serialize_attributes(attributes: Vec<Attribute>) -> PyResult<PyObject>
def serialize_attributes(attributes: List[Attribute]) -> None: ...

# pub fn deserialize_attributes(serialized: &Bound<'_, PyBytes>) -> PyResult<Vec<Attribute>>
def deserialize_attributes(serialized: bytes) -> List[Attribute]: ...

class KvsSetOperation:
    timestamp: int
    ttl: Optional[int]
    attributes: List[Attribute]

class KvsDeleteOperation:
    timestamp: int
    attributes: List[Attribute]

class KvsSubscription:
    def __init__(self, name: str, max_inflight_ops: int): ...
    def recv(self) -> Optional[Union[KvsSetOperation, KvsDeleteOperation]]: ...
    def try_recv(self) -> Optional[Union[KvsSetOperation, KvsDeleteOperation]]: ...
