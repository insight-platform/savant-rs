use crate::utils::python::with_gil;
use pyo3::{Py, PyAny, PyObject};
use std::collections::HashMap;

pub trait PyObjectMeta: Send {
    fn get_py_objects_ref(&self) -> &HashMap<(String, String), PyObject>;
    fn get_py_objects_ref_mut(&mut self) -> &mut HashMap<(String, String), PyObject>;

    fn get_py_object_by_ref(&self, namespace: &str, name: &str) -> Option<Py<PyAny>> {
        with_gil(|py| {
            self.get_py_objects_ref()
                .get(&(namespace.to_owned(), name.to_owned()))
                .map(|o| o.clone_ref(py))
        })
    }

    fn del_py_object(&mut self, namespace: &str, name: &str) -> Option<PyObject> {
        self.get_py_objects_ref_mut()
            .remove(&(namespace.to_owned(), name.to_owned()))
    }

    fn set_py_object(
        &mut self,
        namespace: &str,
        name: &str,
        pyobject: PyObject,
    ) -> Option<PyObject> {
        self.get_py_objects_ref_mut()
            .insert((namespace.to_owned(), name.to_owned()), pyobject)
    }

    fn clear_py_objects(&mut self) {
        self.get_py_objects_ref_mut().clear();
    }

    fn list_py_objects(&self) -> Vec<(String, String)> {
        self.get_py_objects_ref()
            .keys()
            .map(|(namespace, name)| (namespace.clone(), name.clone()))
            .collect()
    }
}
