use hashbrown::HashMap;
use lazy_static::lazy_static;
use log::info;
use parking_lot::Mutex;
use pyo3::prelude::*;
use pyo3::sync::GILOnceCell;
use pyo3::types::{PyDict, PyFunction, PyTuple, PyType};
use pyo3::{PyObject, PyResult};

type GilOnceType = GILOnceCell<Py<PyType>>;
type GilOnceFunc = GILOnceCell<Py<PyFunction>>;

lazy_static! {
    static ref CLASSES: Mutex<HashMap<(String, String), GilOnceType>> = Mutex::new(HashMap::new());
}

#[pyfunction]
pub fn preload_type(py: Python, module_name: &str, class_name: &str) -> PyResult<()> {
    info!("Preloading type {} from module {}", class_name, module_name);
    let mut bind = CLASSES.lock();
    bind.entry((module_name.to_string(), class_name.to_string()))
        .or_default()
        .import(py, module_name, class_name)?;
    Ok(())
}

#[pyclass]
pub struct PyHandler {
    pub instance: PyObject,
}

impl PyHandler {
    pub fn new(
        py: Python,
        module_name: &str,
        class_name: &str,
        element_name: &str,
        args: &str,
    ) -> PyResult<Self> {
        info!("Creating PyHandler");
        let mut bind = CLASSES.lock();
        let handler_type = bind
            .entry((module_name.to_string(), class_name.to_string()))
            .or_default()
            .import(py, module_name, class_name)?;
        info!("Module {} Class {} imported", module_name, class_name);
        static JSON_LOADS: GilOnceFunc = GILOnceCell::new();
        let kwargs_any = JSON_LOADS.import(py, "json", "loads")?.call1((args,))?;
        info!("Kwargs created");
        let kwargs = kwargs_any.downcast::<PyDict>()?;
        info!("Kwargs casted to PyDict");
        let instance = handler_type.call((element_name,), Some(kwargs))?.unbind();
        info!("Handler instance created");
        Ok(Self { instance })
    }

    pub fn call<'py, A>(&self, py: Python<'py>, args: A) -> PyResult<PyObject>
    where
        A: IntoPyObject<'py, Target = PyTuple>,
    {
        self.instance.call1(py, args)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gst::FlowResult;
    use pyo3::ffi::c_str;
    use pyo3::types::PyDict;

    #[test]
    fn test_py_handler() -> PyResult<()> {
        #[pyclass]
        struct MyClass;

        #[pymethods]
        impl MyClass {
            #[new]
            #[pyo3(signature = (element_name, **kwargs))]
            fn new(element_name: &str, kwargs: Option<&Bound<'_, PyDict>>) -> Self {
                assert!(!element_name.is_empty());
                assert!(!kwargs.is_none());
                Self
            }

            fn __call__(&self) -> PyResult<FlowResult> {
                Ok(FlowResult::Ok)
            }
        }

        Python::with_gil(|py| {
            let module = PyModule::new(py, "mymod")?;
            let sys = PyModule::import(py, "sys")?;
            let sys_modules_bind = sys.as_ref().getattr("modules")?;
            let sys_modules = sys_modules_bind.downcast::<PyDict>()?;
            sys_modules.set_item("mymod", &module)?;
            module.add_class::<MyClass>()?;

            let handler = PyHandler::new(py, "mymod", "MyClass", "element", r#"{"object": 1}"#)?;
            let res = handler.call(py, ())?;
            let res = res.extract::<FlowResult>(py)?;
            assert_eq!(res, FlowResult::Ok);

            Ok(())
        })
    }
}
