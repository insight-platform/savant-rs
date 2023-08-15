use crate::with_gil;
use parking_lot::RwLock;
use pyo3::{Py, PyAny, PyObject};
use std::collections::HashMap;
use std::sync::Arc;

pub trait PyObjectMeta: Send {
    fn get_py_objects_ref(&self) -> Arc<RwLock<HashMap<(String, String), PyObject>>>;

    fn get_py_object_by_ref(&self, namespace: &str, name: &str) -> Option<Py<PyAny>> {
        with_gil!(|py| {
            self.get_py_objects_ref()
                .read()
                .get(&(namespace.to_owned(), name.to_owned()))
                .map(|o| o.clone_ref(py))
        })
    }

    fn del_py_object(&self, namespace: &str, name: &str) -> Option<PyObject> {
        self.get_py_objects_ref()
            .write()
            .remove(&(namespace.to_owned(), name.to_owned()))
    }

    fn set_py_object(&self, namespace: &str, name: &str, pyobject: PyObject) -> Option<PyObject> {
        self.get_py_objects_ref()
            .write()
            .insert((namespace.to_owned(), name.to_owned()), pyobject)
    }

    fn clear_py_objects(&self) {
        self.get_py_objects_ref().write().clear();
    }

    fn list_py_objects(&self) -> Vec<(String, String)> {
        self.get_py_objects_ref()
            .read()
            .keys()
            .map(|(namespace, name)| (namespace.clone(), name.clone()))
            .collect()
    }
}
