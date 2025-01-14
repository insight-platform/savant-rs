from typing import List, Optional
from savant_rs.primitives import Attribute


# pub fn set_attributes(attributes: Vec<Attribute>, ttl: Option<u64>)
def set_attributes(attributes: List[Attribute], ttl: Optional[int]) -> None: ...


def search_attributes(ns: Optional[str], name: Optional[str], no_gil: bool) -> List[Attribute]: ...


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
