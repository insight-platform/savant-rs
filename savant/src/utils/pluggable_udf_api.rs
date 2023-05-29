use crate::primitives::Object;
use crate::utils::python::no_gil;
use hashbrown::HashMap;
use libloading::os::unix::Symbol;
use pyo3::prelude::*;
use std::sync::Arc;

pub type UnaryObjectPredicateFunc = extern "C" fn(o: &Object) -> bool;
pub type UnaryObjectPredicate = Symbol<UnaryObjectPredicateFunc>;

pub type BinaryObjectMatchPredicateFunc = extern "C" fn(left: &Object, right: &Object) -> bool;
pub type BinaryObjectMatchPredicate = Symbol<BinaryObjectMatchPredicateFunc>;

#[pyclass]
#[derive(Clone, Debug)]
pub enum UserFunctionKind {
    UnaryObjectPredicate,
    BinaryObjectMatchPredicate,
}

#[derive(Clone, Debug)]
pub enum UserFunction {
    UnaryObjectPredicate(UnaryObjectPredicate),
    BinaryObjectMatchPredicate(BinaryObjectMatchPredicate),
}

#[pyclass]
#[derive(Debug)]
pub struct UserFunctionPluginFactory {
    lib: Option<libloading::Library>,
    functions: Option<HashMap<String, UserFunction>>,
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct UserFunctionPlugin {
    inner: Arc<UserFunctionPluginFactory>,
}

impl UserFunctionPlugin {
    pub fn eval(&self, name: &str, args: &[&Object]) -> anyhow::Result<bool> {
        let inner = self.inner.clone();
        let func = match inner.functions.as_ref().unwrap().get(name) {
            Some(func) => func,
            None => panic!("Function {} not found", name),
        };

        match func {
            UserFunction::UnaryObjectPredicate(f) => {
                let arg = args.get(0).expect("Unary predicate requires one argument");
                Ok(f(arg))
            }
            UserFunction::BinaryObjectMatchPredicate(f) => {
                let left = args
                    .get(0)
                    .expect("Binary predicate requires two arguments");
                let right = args
                    .get(1)
                    .expect("Binary predicate requires two arguments");
                Ok(f(left, right))
            }
        }
    }
}

#[pymethods]
impl UserFunctionPlugin {
    #[pyo3(name = "eval")]
    fn eval_py(&self, name: String, args: Vec<Object>) -> PyResult<bool> {
        no_gil(|| {
            self.eval(&name, args.iter().map(|e| e).collect::<Vec<_>>().as_slice())
                .map_err(|e| {
                    pyo3::exceptions::PyException::new_err(format!(
                        "Error evaluating function: {}",
                        e
                    ))
                })
        })
    }
}

impl UserFunctionPluginFactory {
    pub fn new(name: &str) -> anyhow::Result<Self> {
        let lib = unsafe { libloading::Library::new(name)? };
        let functions = Some(HashMap::new());
        Ok(Self {
            lib: Some(lib),
            functions,
        })
    }

    pub fn register_function(&mut self, name: &str, kind: UserFunctionKind) -> anyhow::Result<()> {
        let byte_name = name.as_bytes();
        let lib = self.lib.as_ref().unwrap();
        let func = match kind {
            UserFunctionKind::UnaryObjectPredicate => unsafe {
                let func: libloading::Symbol<UnaryObjectPredicateFunc> = lib.get(byte_name)?;
                UserFunction::UnaryObjectPredicate(func.into_raw())
            },
            UserFunctionKind::BinaryObjectMatchPredicate => unsafe {
                let func: libloading::Symbol<BinaryObjectMatchPredicateFunc> =
                    lib.get(byte_name)?;
                UserFunction::BinaryObjectMatchPredicate(func.into_raw())
            },
        };
        if let Some(f) = self.functions.as_mut() {
            f.insert(name.to_string(), func);
        }
        Ok(())
    }
}

#[pymethods]
impl UserFunctionPluginFactory {
    #[new]
    pub fn init(name: String) -> PyResult<Self> {
        no_gil(|| Self::new(&name)).map_err(|e| {
            pyo3::exceptions::PyException::new_err(format!("Error initializing plugin: {}", e))
        })
    }

    #[pyo3(name = "register_function")]
    pub fn register_function_py(&mut self, name: String, kind: UserFunctionKind) -> PyResult<()> {
        no_gil(|| self.register_function(&name, kind)).map_err(|e| {
            pyo3::exceptions::PyException::new_err(format!("Error registering function: {}", e))
        })
    }

    pub fn initialize(&mut self) -> UserFunctionPlugin {
        if self.lib.is_none() {
            panic!("UserFunctionPlugin already initialized");
        }

        UserFunctionPlugin {
            inner: Arc::new(Self {
                lib: self.lib.take(),
                functions: self.functions.take(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test::utils::gen_object;

    #[test]
    fn test_plugin_api_2() -> anyhow::Result<()> {
        pyo3::prepare_freethreaded_python();
        let mut p = UserFunctionPluginFactory::new("../target/release/libsample_plugin.so")?;
        p.register_function(
            "binary_op_parent",
            UserFunctionKind::BinaryObjectMatchPredicate,
        )?;
        p.register_function("unary_op_even", UserFunctionKind::UnaryObjectPredicate)?;
        let p = p.initialize();

        let o = gen_object(12);
        assert!(p.eval("unary_op_even", &[&o])?);

        let o = gen_object(13);
        assert!(!p.eval("unary_op_even", &[&o])?);

        assert!(!p.eval("binary_op_parent", &[&o, &o])?);

        let o2 = gen_object(12);
        o.set_parent(Some(o2.clone()));
        assert!(p.eval("binary_op_parent", &[&o, &o2])?);

        Ok(())
    }
}
