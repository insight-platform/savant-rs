use anyhow::{anyhow, bail, Result};
use etcd_dynamic_state::etcd_api::{EtcdClient, VarPathSpec};
use etcd_dynamic_state::parameter_storage::EtcdParameterStorage;
use evalexpr::*;
use hashbrown::HashMap;
use lazy_static::lazy_static;
use parking_lot::Mutex;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::env;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::runtime::Runtime;

lazy_static! {
    static ref RESOLVERS: Mutex<HashMap<String, Arc<dyn SymbolResolver>>> = {
        let utility_resolver = Arc::new(UtilityResolver);
        let name = utility_resolver.name().to_string();

        Mutex::new(HashMap::from([(
            name,
            utility_resolver as Arc<dyn SymbolResolver>,
        )]))
    };
}

pub fn register_symbol_resolver(resolver: Arc<dyn SymbolResolver>) {
    let name = resolver.name().to_string();
    RESOLVERS.lock().insert(name, resolver);
}

pub(crate) fn get_symbol_resolver(name: &str) -> Option<Arc<dyn SymbolResolver>> {
    RESOLVERS.lock().get(name).cloned()
}

#[pyfunction]
pub fn register_env_symbol_resolver() {
    register_symbol_resolver(Arc::new(EnvSymbolResolver) as Arc<dyn SymbolResolver>);
}

#[pyfunction]
pub fn register_config_symbol_resolver(symbols: std::collections::HashMap<String, String>) {
    let mut resolver = ConfigSymbolResolver::new();
    for (key, value) in symbols {
        resolver.add_symbol(key, value);
    }
    register_symbol_resolver(Arc::new(resolver) as Arc<dyn SymbolResolver>);
}

#[pyfunction]
#[pyo3(signature = (hosts = vec!["127.0.0.1:2379".to_string()], credentials = None, watch_path = "savant".to_string(), connect_timeout = 5, watch_path_wait_timeout = 5))]
pub fn register_etcd_symbol_resolver(
    hosts: Vec<String>,
    credentials: Option<(String, String)>,
    watch_path: String,
    connect_timeout: u64,
    watch_path_wait_timeout: u64,
) -> PyResult<()> {
    let resolver = EtcdSymbolResolver::new(
        hosts,
        credentials,
        watch_path,
        connect_timeout,
        watch_path_wait_timeout,
    )
    .map_err(|e| {
        PyRuntimeError::new_err(format!("Failed to create etcd symbol resolver: {}", e))
    })?;

    register_symbol_resolver(Arc::new(resolver) as Arc<dyn SymbolResolver>);

    Ok(())
}

const ENV_FUNC: &'static str = "env";
const ETCD_FUNC: &'static str = "etcd";
const CONFIG_FUNC: &'static str = "config";

fn cast_str_to_primitive_type(s: &str, value: &Value) -> Result<Value> {
    match value {
        Value::String(_) => Ok(Value::String(s.to_string())),
        Value::Float(_) => Ok(Value::Float(s.parse()?)),
        Value::Int(_) => Ok(Value::Int(s.parse()?)),
        Value::Boolean(_) => Ok(Value::Boolean(s.parse()?)),
        _ => bail!("env: the value must be a string, float, int or boolean"),
    }
}

pub trait SymbolResolver: Send + Sync {
    fn resolve(&self, func: &str, expr: &Value) -> Result<Value>;
    fn exported_symbols(&self) -> Vec<&'static str>;
    fn name(&self) -> &'static str;
}

struct EnvSymbolResolver;

impl SymbolResolver for EnvSymbolResolver {
    fn resolve(&self, func: &str, expr: &Value) -> Result<Value> {
        match func {
            "env" => {
                if !expr.is_tuple() {
                    bail!("The function must be called as env(key, default)");
                }
                match expr.as_tuple().unwrap().as_slice() {
                    [Value::String(key), default] => match env::var(key) {
                        Ok(value) => cast_str_to_primitive_type(&value, default),
                        Err(_) => Ok(default.clone()),
                    },
                    _ => unreachable!(),
                }
            }
            _ => bail!("unknown function: {} called for {:?}", func, expr),
        }
    }

    fn exported_symbols(&self) -> Vec<&'static str> {
        vec![ENV_FUNC]
    }

    fn name(&self) -> &'static str {
        "EnvSymbolResolver"
    }
}

struct EtcdSymbolResolver {
    inner: Arc<Mutex<EtcdParameterStorage>>,
    runtime: Option<Arc<Runtime>>,
    prefix: String,
}

impl EtcdSymbolResolver {
    pub fn new(
        hosts: Vec<String>,
        credentials: Option<(String, String)>,
        watch_path: String,
        connect_timeout: u64,
        watch_path_wait_timeout: u64,
    ) -> Result<Self> {
        assert!(watch_path_wait_timeout > 0);
        assert!(connect_timeout > 0);

        let runtime = Runtime::new().unwrap();
        let client = EtcdClient::new(hosts, credentials, watch_path.clone(), 60, connect_timeout);

        let client = runtime.block_on(client)?;

        let mut parameter_storage = EtcdParameterStorage::with_client(client);
        parameter_storage.run(&runtime)?;
        parameter_storage.order_data_update(VarPathSpec::Prefix(watch_path.clone()))?;
        _ = parameter_storage.wait_for_key(&watch_path, watch_path_wait_timeout * 1000); // wait for the first update

        Ok(Self {
            inner: Arc::new(Mutex::new(parameter_storage)),
            runtime: Some(Arc::new(runtime)),
            prefix: watch_path,
        })
    }

    fn _shutdown(&mut self) -> Result<()> {
        let rt = self
            .runtime
            .take()
            .ok_or_else(|| anyhow!("EtcdParameterStorage has already been stopped"))?;

        let rt = Arc::try_unwrap(rt).map_err(|survived_rt| {
                self.runtime = Some(survived_rt);
                anyhow!(
                    "Failed to destroy EtcdParameterStorage: there are more than one references to the runtime."
                )
            })?;

        self.inner.lock().stop(rt)?;

        Ok(())
    }

    fn get_data(&self, key: &str) -> Result<Option<String>> {
        let mut path = PathBuf::from(&self.prefix);
        let key_path = Path::new(key);
        if key_path.is_absolute() {
            bail!("key must be relative to prefix {}", self.prefix);
        }
        path.push(key_path);
        let path = path.to_str().unwrap();
        let data_opt = self.inner.lock().get_data(path)?;

        match data_opt {
            Some((_crc, data)) => Ok(Some(String::from_utf8_lossy(&data).to_string())),
            None => Ok(None),
        }
    }

    fn is_active(&self) -> bool {
        self.inner.lock().is_active()
    }
}

impl Drop for EtcdSymbolResolver {
    fn drop(&mut self) {
        if self.is_active() {
            self._shutdown()
                .expect("Failed to shutdown EtcdParameterStorage");
        }
    }
}

impl SymbolResolver for EtcdSymbolResolver {
    fn resolve(&self, func: &str, expr: &Value) -> Result<Value> {
        match func {
            "etcd" => {
                if !expr.is_tuple() {
                    bail!("The function must be called as etcd(key, default)");
                }

                match expr.as_tuple().unwrap().as_slice() {
                    [Value::String(key), default] => match self.get_data(key)? {
                        Some(value) => cast_str_to_primitive_type(&value, default),
                        None => Ok(default.clone()),
                    },
                    _ => unreachable!(),
                }
            }
            _ => bail!("unknown function: {} called for {:?}", func, expr),
        }
    }

    fn exported_symbols(&self) -> Vec<&'static str> {
        vec![ETCD_FUNC]
    }

    fn name(&self) -> &'static str {
        "EtcdSymbolResolver"
    }
}

struct ConfigSymbolResolver {
    symbols: hashbrown::HashMap<String, String>,
}

impl ConfigSymbolResolver {
    pub fn new() -> Self {
        Self {
            symbols: hashbrown::HashMap::new(),
        }
    }

    pub fn add_symbol(&mut self, name: String, value: String) {
        self.symbols.insert(name, value);
    }
}

impl SymbolResolver for ConfigSymbolResolver {
    fn resolve(&self, func: &str, expr: &Value) -> Result<Value> {
        match func {
            "config" => {
                if !expr.is_tuple() {
                    bail!("The function must be called as config(key, default)");
                }
                match expr.as_tuple().unwrap().as_slice() {
                    [Value::String(key), default] => match self.symbols.get(key) {
                        Some(value) => cast_str_to_primitive_type(value, default),
                        None => Ok(default.clone()),
                    },
                    _ => unreachable!(),
                }
            }
            _ => bail!("unknown function: {} called for {:?}", func, expr),
        }
    }

    fn exported_symbols(&self) -> Vec<&'static str> {
        vec![CONFIG_FUNC]
    }

    fn name(&self) -> &'static str {
        "ConfigResolver"
    }
}

struct UtilityResolver;

impl SymbolResolver for UtilityResolver {
    fn resolve(&self, func: &str, expr: &Value) -> Result<Value> {
        match func {
            "is_boolean" => Ok(Value::Boolean(expr.is_boolean())),
            "is_float" => Ok(Value::Boolean(expr.is_float())),
            "is_int" => Ok(Value::Boolean(expr.is_int())),
            "is_string" => Ok(Value::Boolean(expr.is_string())),
            "is_tuple" => Ok(Value::Boolean(expr.is_tuple())),
            "is_empty" => Ok(Value::Boolean(expr.is_empty())),
            _ => bail!("unknown function: {} called for {:?}", func, expr),
        }
    }

    fn exported_symbols(&self) -> Vec<&'static str> {
        vec![
            "is_boolean",
            "is_float",
            "is_int",
            "is_string",
            "is_tuple",
            "is_empty",
        ]
    }

    fn name(&self) -> &'static str {
        "UtilityResolver"
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::eval_resolvers::{
        cast_str_to_primitive_type, EnvSymbolResolver, SymbolResolver,
    };
    use evalexpr::Value;
    use std::env;

    #[test]
    fn test_conversions() {
        let default = Value::Boolean(false);
        let r = cast_str_to_primitive_type("true", &default).unwrap();
        assert_eq!(r, Value::Boolean(true));

        let default = Value::String("hello".to_string());
        let r = cast_str_to_primitive_type("world", &default).unwrap();
        assert_eq!(r, Value::String("world".to_string()));

        let default = Value::Int(0);
        let r = cast_str_to_primitive_type("123", &default).unwrap();
        assert_eq!(r, Value::Int(123));

        let default = Value::Float(0.0);
        let r = cast_str_to_primitive_type("123.456", &default).unwrap();
        assert_eq!(r, Value::Float(123.456));
    }

    #[test]
    fn test_env_resolver() {
        let resolver = EnvSymbolResolver;
        let default = Value::String("".to_string());
        let value = resolver
            .resolve(
                "env",
                &Value::Tuple(vec![Value::String("PATH".to_string()), default]),
            )
            .unwrap();

        assert_eq!(value, Value::String(env::var("PATH").unwrap()));

        let default = Value::String("DEFAULT".to_string());
        let value = resolver
            .resolve(
                "env",
                &Value::Tuple(vec![Value::String("UNKNOWN".to_string()), default.clone()]),
            )
            .unwrap();

        assert_eq!(value, default);
    }

    #[test]
    fn test_config_resolver() {
        let mut resolver = crate::utils::eval_resolvers::ConfigSymbolResolver::new();
        resolver.add_symbol("config.key".to_string(), "value".to_string());
        let default = Value::String("".to_string());
        let value = resolver
            .resolve(
                "config",
                &Value::Tuple(vec![Value::String("config.key".to_string()), default]),
            )
            .unwrap();

        assert_eq!(value, Value::String("value".to_string()));

        let default = Value::String("DEFAULT".to_string());
        let value = resolver
            .resolve(
                "config",
                &Value::Tuple(vec![Value::String("UNKNOWN".to_string()), default.clone()]),
            )
            .unwrap();

        assert_eq!(value, default);
    }

    #[test]
    fn test_utility_resolver() {
        let resolver = crate::utils::eval_resolvers::UtilityResolver;
        let value = resolver
            .resolve("is_boolean", &Value::Boolean(true))
            .unwrap();
        assert_eq!(value, Value::Boolean(true));

        let value = resolver.resolve("is_float", &Value::Float(1.0)).unwrap();
        assert_eq!(value, Value::Boolean(true));
        let value = resolver.resolve("is_float", &Value::Boolean(true)).unwrap();
        assert_eq!(value, Value::Boolean(false));

        let value = resolver.resolve("is_int", &Value::Int(1)).unwrap();
        assert_eq!(value, Value::Boolean(true));
        let value = resolver.resolve("is_int", &Value::Boolean(true)).unwrap();
        assert_eq!(value, Value::Boolean(false));

        let value = resolver
            .resolve("is_string", &Value::String("hello".to_string()))
            .unwrap();
        assert_eq!(value, Value::Boolean(true));
        let value = resolver
            .resolve("is_string", &Value::Boolean(true))
            .unwrap();
        assert_eq!(value, Value::Boolean(false));

        let value = resolver.resolve("is_tuple", &Value::Tuple(vec![])).unwrap();
        assert_eq!(value, Value::Boolean(true));

        let value = resolver.resolve("is_empty", &Value::Empty).unwrap();
        assert_eq!(value, Value::Boolean(true));
    }

    #[test]
    #[ignore]
    fn test_etcd_resolver() {
        let resolver = crate::utils::eval_resolvers::EtcdSymbolResolver::new(
            vec!["127.0.0.1:2379".to_string()],
            None,
            "savant".to_string(),
            1,
            1,
        )
        .unwrap();
        let default = Value::Int(0);
        let value = resolver
            .resolve(
                "etcd",
                &Value::Tuple(vec![Value::String("abc".to_string()), default]),
            )
            .unwrap();
        assert_eq!(value, Value::Int(1));

        let default = Value::Int(-1);
        let value = resolver
            .resolve(
                "etcd",
                &Value::Tuple(vec![Value::String("xyz".to_string()), default]),
            )
            .unwrap();
        assert_eq!(value, Value::Int(-1));
    }
}
