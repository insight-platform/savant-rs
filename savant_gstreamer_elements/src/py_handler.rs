use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use pyo3::{PyObject, PyResult};

pub struct PyHandler {
    pub instance: PyObject,
}

impl PyHandler {
    pub fn new(py: Python, module_name: &str, class_name: &str, args: &str) -> PyResult<Self> {
        let module = PyModule::import(py, module_name)?;
        let json_module = PyModule::import(py, "json")?;
        let json = json_module.getattr("loads")?;
        let args_binding = json.call1((args,))?;
        let args = args_binding.downcast::<PyDict>()?;
        let class = module.getattr(class_name)?;
        let instance = class.call((), Some(args))?.unbind();
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
    use pyo3::types::PyDict;
    use savant_core_py::gst::FlowResult;

    #[test]
    fn test_py_handler() -> PyResult<()> {
        #[pyclass]
        struct MyClass;

        #[pymethods]
        impl MyClass {
            #[new]
            #[pyo3(signature = (**kwargs))]
            fn new(kwargs: Option<&Bound<'_, PyDict>>) -> Self {
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

            let handler = PyHandler::new(py, "mymod", "MyClass", r#"{"object": 1}"#)?;
            let res = handler.call(py, ())?;
            let res = res.extract::<FlowResult>(py)?;
            assert_eq!(res, FlowResult::Ok);

            Ok(())
        })
    }
}
