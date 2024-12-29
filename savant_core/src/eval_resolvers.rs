pub const ENV_FUNC: &str = "env";
pub const CONFIG_FUNC: &str = "config";
const ETCD_FUNC: &str = "etcd";

pub use resolvers::{EtcdCredentials, EvalWithResolvers, TlsConfig};
pub use singleton::*;
pub use utils::*;

mod utils {
    use anyhow::{bail, Result};
    use evalexpr::Value;

    pub fn cast_str_to_primitive_type(s: &str, value: &Value) -> Result<Value> {
        match value {
            Value::String(_) => Ok(Value::String(s.to_string())),
            Value::Float(_) => Ok(Value::Float(s.parse()?)),
            Value::Int(_) => Ok(Value::Int(s.parse()?)),
            Value::Boolean(_) => Ok(Value::Boolean(s.parse()?)),
            _ => bail!("env: the value must be a string, float, int or boolean"),
        }
    }

    #[inline(always)]
    pub fn utility_resolver_name() -> &'static str {
        "utility-resolver"
    }

    #[inline(always)]
    pub fn env_resolver_name() -> &'static str {
        "env-resolver"
    }

    #[inline(always)]
    pub fn config_resolver_name() -> &'static str {
        "config-resolver"
    }

    #[inline(always)]
    pub fn etcd_resolver_name() -> &'static str {
        "etcd-resolver"
    }
}

pub(crate) mod resolvers {
    use crate::eval_resolvers::{
        cast_str_to_primitive_type, config_resolver_name, env_resolver_name, etcd_resolver_name,
        get_symbol_resolver, utility_resolver_name, CONFIG_FUNC, ENV_FUNC, ETCD_FUNC,
    };
    use crate::{get_or_init_async_runtime, trace};
    use anyhow::{bail, Result};
    use etcd_client::{Certificate, Identity, TlsOptions};
    use etcd_dynamic_state::etcd_api::{EtcdClient, VarPathSpec};
    use etcd_dynamic_state::parameter_storage::EtcdParameterStorage;
    use evalexpr::{EvalexprError, EvalexprResult, Value};
    use hashbrown::HashMap;
    use parking_lot::{Mutex, RwLock};
    use std::any::Any;
    use std::env;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    pub trait EvalWithResolvers {
        fn get_resolvers(&self) -> &'_ [String];

        fn resolve(&self, identifier: &str, argument: &Value) -> EvalexprResult<Value> {
            let res = get_symbol_resolver(identifier);
            match res {
                Some((r, executor)) => {
                    if self.get_resolvers().contains(&r) {
                        executor
                            .resolve(identifier, argument)
                            .map_err(|e| EvalexprError::CustomMessage(e.to_string()))
                    } else {
                        Err(EvalexprError::FunctionIdentifierNotFound(
                            identifier.to_string(),
                        ))
                    }
                }
                None => Err(EvalexprError::FunctionIdentifierNotFound(
                    identifier.to_string(),
                )),
            }
        }
    }

    pub trait SymbolResolver: Send + Sync {
        fn resolve(&self, func: &str, expr: &Value) -> Result<Value>;
        fn exported_symbols(&self) -> Vec<&'static str>;
        fn name(&self) -> &'static str;
        fn as_any(&self) -> &dyn Any;
    }

    pub struct EnvSymbolResolver;

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
            env_resolver_name()
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    pub struct ConfigSymbolResolver {
        pub(super) symbols: RwLock<HashMap<String, String>>,
    }

    impl ConfigSymbolResolver {
        pub fn new() -> Self {
            Self {
                symbols: RwLock::new(HashMap::new()),
            }
        }

        pub fn add_symbol(&mut self, name: String, value: String) {
            trace!(self.symbols.write()).insert(name, value);
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
                        [Value::String(key), default] => {
                            match trace!(self.symbols.read_recursive()).get(key) {
                                Some(value) => cast_str_to_primitive_type(value, default),
                                None => Ok(default.clone()),
                            }
                        }
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
            config_resolver_name()
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    pub struct UtilityResolver;

    impl SymbolResolver for UtilityResolver {
        fn resolve(&self, func: &str, expr: &Value) -> Result<Value> {
            match func {
                "is_boolean" => Ok(Value::Boolean(expr.is_boolean())),
                "is_float" => Ok(Value::Boolean(expr.is_float())),
                "is_int" => Ok(Value::Boolean(expr.is_int())),
                "is_string" => Ok(Value::Boolean(expr.is_string())),
                "is_tuple" => Ok(Value::Boolean(expr.is_tuple())),
                "is_empty" => Ok(Value::Boolean(expr.is_empty())),
                "ends_with" => {
                    if !expr.is_tuple() {
                        bail!("The function must be called as ends_with(string, suffix)");
                    }
                    match expr.as_tuple().unwrap().as_slice() {
                        [Value::String(string), Value::String(suffix)] => {
                            Ok(Value::Boolean(string.ends_with(suffix)))
                        }
                        _ => unreachable!(),
                    }
                }
                "starts_with" => {
                    if !expr.is_tuple() {
                        bail!("The function must be called as starts_with(string, prefix)");
                    }
                    match expr.as_tuple().unwrap().as_slice() {
                        [Value::String(string), Value::String(prefix)] => {
                            Ok(Value::Boolean(string.starts_with(prefix)))
                        }
                        _ => unreachable!(),
                    }
                }
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
                "ends_with",
                "starts_with",
            ]
        }

        fn name(&self) -> &'static str {
            utility_resolver_name()
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    pub struct EtcdSymbolResolver {
        inner: Arc<Mutex<EtcdParameterStorage>>,
        prefix: String,
    }

    #[derive(Clone)]
    pub struct EtcdCredentials {
        pub username: String,
        pub password: String,
    }

    #[derive(Clone)]
    pub struct TlsConfig {
        pub ca_cert: String,
        pub client_cert: String,
        pub client_key: String,
    }

    impl EtcdSymbolResolver {
        pub fn new(
            hosts: &[&str],
            credentials: &Option<EtcdCredentials>,
            tls_config: &Option<TlsConfig>,
            watch_path: &str,
            connect_timeout: u64,
            watch_path_wait_timeout: u64,
        ) -> Result<Self> {
            assert!(watch_path_wait_timeout > 0);
            assert!(connect_timeout > 0);

            let runtime = get_or_init_async_runtime();
            let credentials = credentials
                .as_ref()
                .map(|creds| (creds.username.as_str(), creds.password.as_str()));
            let tls = tls_config.as_ref().map(|tls| {
                TlsOptions::default()
                    .ca_certificate(Certificate::from_pem(tls.ca_cert.as_bytes()))
                    .identity(Identity::from_pem(
                        tls.client_cert.as_bytes(),
                        tls.client_key.as_bytes(),
                    ))
            });
            let client =
                EtcdClient::new_with_tls(hosts, &credentials, watch_path, 60, connect_timeout, tls);

            let client = runtime.block_on(client)?;

            let mut parameter_storage = EtcdParameterStorage::with_client(client);
            parameter_storage.run(&runtime)?;
            parameter_storage.order_data_update(VarPathSpec::Prefix(watch_path.to_string()))?;
            _ = parameter_storage.wait_for_key(watch_path, watch_path_wait_timeout * 1000); // wait for the first update

            Ok(Self {
                inner: Arc::new(Mutex::new(parameter_storage)),
                prefix: watch_path.to_string(),
            })
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
            etcd_resolver_name()
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }
}

pub(crate) mod singleton {
    use crate::eval_resolvers::config_resolver_name;
    use crate::eval_resolvers::resolvers::{
        ConfigSymbolResolver, EnvSymbolResolver, EtcdCredentials, EtcdSymbolResolver,
        SymbolResolver, TlsConfig, UtilityResolver,
    };
    use crate::rwlock::SavantRwLock;
    use crate::trace;
    use anyhow::Result;
    use hashbrown::HashMap;
    use lazy_static::lazy_static;
    use std::sync::Arc;

    pub type ResolverValue = (String, Arc<dyn SymbolResolver>);

    lazy_static! {
        static ref RESOLVERS: SavantRwLock<HashMap<String, ResolverValue>> =
            SavantRwLock::new(HashMap::default());
    }

    pub fn register_symbol_resolver(resolver: Arc<dyn SymbolResolver>) {
        let name = resolver.name().to_string();
        let symbols = resolver.exported_symbols();
        let mut r = RESOLVERS.write();
        for s in symbols {
            r.insert(s.to_string(), (name.clone(), resolver.clone()));
        }
        r.insert(name.clone(), (name.clone(), resolver));
    }

    pub fn get_symbol_resolver(symbol: &str) -> Option<ResolverValue> {
        RESOLVERS.read().get(symbol).cloned()
    }

    pub fn register_utility_resolver() {
        register_symbol_resolver(Arc::new(UtilityResolver) as Arc<dyn SymbolResolver>);
    }

    pub fn register_env_resolver() {
        register_symbol_resolver(Arc::new(EnvSymbolResolver) as Arc<dyn SymbolResolver>);
    }

    pub fn register_etcd_resolver(
        hosts: &[&str],
        credentials: &Option<EtcdCredentials>,
        tls_config: &Option<TlsConfig>,
        watch_path: &str,
        connect_timeout: u64,
        watch_path_wait_timeout: u64,
    ) -> Result<()> {
        let resolver = EtcdSymbolResolver::new(
            hosts,
            credentials,
            tls_config,
            watch_path,
            connect_timeout,
            watch_path_wait_timeout,
        )?;

        register_symbol_resolver(Arc::new(resolver) as Arc<dyn SymbolResolver>);

        Ok(())
    }

    pub fn register_config_resolver(symbols: HashMap<String, String>) {
        let mut resolver = ConfigSymbolResolver::new();
        for (key, value) in symbols {
            resolver.add_symbol(key, value);
        }
        register_symbol_resolver(Arc::new(resolver) as Arc<dyn SymbolResolver>);
    }

    pub fn update_config_resolver(symbols: HashMap<String, String>) {
        let r = RESOLVERS.read();
        let resolver = r.get(config_resolver_name());
        if let Some((_, resolver)) = resolver {
            let resolver = resolver
                .as_any()
                .downcast_ref::<ConfigSymbolResolver>()
                .expect("Wrong downcast");

            let mut old_symbols = trace!(resolver.symbols.write());
            old_symbols.extend(symbols);
        } else {
            drop(r);
            register_config_resolver(symbols);
        };
    }

    pub fn unregister_resolver(name: &str) {
        let mut r = RESOLVERS.write();
        let resolver_opt = r.remove(name);
        if let Some((_, resolver)) = resolver_opt {
            let symbols = resolver.exported_symbols();
            for s in symbols {
                r.remove(s);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::eval_resolvers::resolvers::{
        ConfigSymbolResolver, EnvSymbolResolver, EtcdSymbolResolver, SymbolResolver,
        UtilityResolver,
    };
    use crate::eval_resolvers::{
        cast_str_to_primitive_type, config_resolver_name, get_symbol_resolver,
        register_config_resolver, unregister_resolver, update_config_resolver, CONFIG_FUNC,
        ENV_FUNC, ETCD_FUNC,
    };
    use crate::get_or_init_async_runtime;
    use bollard::container::{
        Config, CreateContainerOptions, RemoveContainerOptions, StartContainerOptions,
    };
    use bollard::image::CreateImageOptions;
    use bollard::models::{HostConfig, PortBinding};
    use bollard::Docker;
    use etcd_dynamic_state::etcd_api::{EtcdClient, Operation};
    use evalexpr::Value;
    use futures_util::TryStreamExt;
    use hashbrown::HashMap;
    use std::env;
    use std::time::Duration;

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
                ENV_FUNC,
                &Value::Tuple(vec![Value::String("PATH".to_string()), default]),
            )
            .unwrap();

        assert_eq!(value, Value::String(env::var("PATH").unwrap()));

        let default = Value::String("DEFAULT".to_string());
        let value = resolver
            .resolve(
                ENV_FUNC,
                &Value::Tuple(vec![Value::String("UNKNOWN".to_string()), default.clone()]),
            )
            .unwrap();

        assert_eq!(value, default);
    }

    #[test]
    fn test_config_resolver() {
        let mut resolver = ConfigSymbolResolver::new();
        resolver.add_symbol("config.key".to_string(), "value".to_string());
        let default = Value::String("".to_string());
        let value = resolver
            .resolve(
                CONFIG_FUNC,
                &Value::Tuple(vec![Value::String("config.key".to_string()), default]),
            )
            .unwrap();

        assert_eq!(value, Value::String("value".to_string()));

        let default = Value::String("DEFAULT".to_string());
        let value = resolver
            .resolve(
                CONFIG_FUNC,
                &Value::Tuple(vec![Value::String("UNKNOWN".to_string()), default.clone()]),
            )
            .unwrap();

        assert_eq!(value, default);
    }

    #[test]
    fn test_update_config_resolver() {
        register_config_resolver(HashMap::from([(
            String::from("key"),
            String::from("value"),
        )]));
        let (_, res) = get_symbol_resolver(CONFIG_FUNC).unwrap();
        let default = Value::String("".to_string());
        let value = res
            .resolve(
                CONFIG_FUNC,
                &Value::Tuple(vec![Value::String("key".to_string()), default]),
            )
            .unwrap();

        assert_eq!(value, Value::String("value".to_string()));

        update_config_resolver(HashMap::from([(
            String::from("key"),
            String::from("value2"),
        )]));

        let (_, res) = get_symbol_resolver(CONFIG_FUNC).unwrap();
        let default = Value::String("".to_string());

        let value = res
            .resolve(
                CONFIG_FUNC,
                &Value::Tuple(vec![Value::String("key".to_string()), default]),
            )
            .unwrap();

        assert_eq!(value, Value::String("value2".to_string()));

        unregister_resolver(config_resolver_name());
        let resolver_opt = get_symbol_resolver(CONFIG_FUNC);
        assert!(resolver_opt.is_none());
    }

    #[test]
    fn test_utility_resolver() {
        let resolver = UtilityResolver;
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
    #[serial_test::serial]
    fn test_etcd_resolver() -> anyhow::Result<()> {
        let runtime = get_or_init_async_runtime();
        let docker = Docker::connect_with_local_defaults()?;
        let ccr: anyhow::Result<String> = runtime.block_on(async {
            let name = "test-etcd-no-tls";
            let _ = docker
                .remove_container(
                    name,
                    Some(RemoveContainerOptions {
                        force: true,
                        ..Default::default()
                    }),
                )
                .await;

            const IMAGE: &str = "bitnami/etcd:latest";

            docker
                .create_image(
                    Some(CreateImageOptions {
                        from_image: IMAGE,
                        ..Default::default()
                    }),
                    None,
                    None,
                )
                .try_collect::<Vec<_>>()
                .await?;

            let ccr = docker
                .create_container(
                    Some(CreateContainerOptions {
                        name: name.to_string(),
                        platform: None,
                    }),
                    Config {
                        image: Some(IMAGE.to_string()),
                        env: Some(vec!["ALLOW_NONE_AUTHENTICATION=yes".to_string()]),
                        host_config: Some(HostConfig {
                            port_bindings: Some(
                                vec![(
                                    "2379/tcp".to_string(),
                                    Some(vec![PortBinding {
                                        host_ip: None,
                                        host_port: Some("22379".to_string()),
                                    }]),
                                )]
                                .into_iter()
                                .collect(),
                            ),
                            ..Default::default()
                        }),
                        ..Default::default()
                    },
                )
                .await?;

            docker
                .start_container(name, None::<StartContainerOptions<String>>)
                .await?;

            let mut client;
            let mut max_retries = 10;
            loop {
                let c = EtcdClient::new(&["127.0.0.1:22379"], &None, "savant", 5, 20).await;
                if c.is_ok() {
                    client = c?;
                    break;
                }
                tokio::time::sleep(Duration::from_secs(5)).await;
                max_retries -= 1;
                if max_retries == 0 {
                    panic!("Failed to connect to Etcd in {}", max_retries);
                }
            }

            client
                .kv_operations(vec![
                    Operation::Set {
                        key: "savant/abc".into(),
                        value: "1".into(),
                        with_lease: false,
                    },
                    Operation::Set {
                        key: "savant/xyz".into(),
                        value: "-1".into(),
                        with_lease: false,
                    },
                ])
                .await?;
            let id = ccr.id.clone();
            Ok(id)
        });

        let ccr = ccr?;

        let resolver = EtcdSymbolResolver::new(&["127.0.0.1:22379"], &None, &None, "savant", 1, 1)?;
        let default = Value::Int(0);
        let value = resolver.resolve(
            ETCD_FUNC,
            &Value::Tuple(vec![Value::String("abc".to_string()), default]),
        )?;
        assert_eq!(value, Value::Int(1));

        let default = Value::Int(-1);
        let value = resolver.resolve(
            ETCD_FUNC,
            &Value::Tuple(vec![Value::String("xyz".to_string()), default]),
        )?;
        assert_eq!(value, Value::Int(-1));

        runtime.block_on(async {
            docker
                .remove_container(
                    &ccr,
                    Some(RemoveContainerOptions {
                        force: true,
                        ..Default::default()
                    }),
                )
                .await?;
            Ok(())
        })
    }
}
