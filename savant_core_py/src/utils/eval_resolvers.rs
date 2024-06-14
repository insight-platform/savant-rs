use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::collections::HashMap;

/// Returns the system name of utility resolver.
///
/// Returns
/// -------
/// str
///   The name of the utility resolver.
///
#[pyfunction]
pub fn utility_resolver_name() -> &'static str {
    savant_core::eval_resolvers::utility_resolver_name()
}

/// Returns the system name of env resolver.
///
/// Returns
/// -------
/// str
///   The name of the env resolver.
///
#[pyfunction]
pub fn env_resolver_name() -> &'static str {
    savant_core::eval_resolvers::env_resolver_name()
}

/// Returns the system name of config resolver.
///
/// Returns
/// -------
/// str
///   The name of the config resolver.
///
#[pyfunction]
pub fn config_resolver_name() -> &'static str {
    savant_core::eval_resolvers::config_resolver_name()
}

/// Returns the system name of etcd resolver.
///
/// Returns
/// -------
/// str
///   The name of the etcd resolver.
///
#[pyfunction]
pub fn etcd_resolver_name() -> &'static str {
    savant_core::eval_resolvers::etcd_resolver_name()
}

#[derive(Clone)]
#[pyclass]
pub struct EtcdCredentials(savant_core::eval_resolvers::EtcdCredentials);

#[pymethods]
impl EtcdCredentials {
    #[new]
    fn new(username: String, password: String) -> Self {
        EtcdCredentials(savant_core::eval_resolvers::EtcdCredentials { username, password })
    }
}

#[derive(Clone)]
#[pyclass]
pub struct TlsConfig(savant_core::eval_resolvers::TlsConfig);

#[pymethods]
impl TlsConfig {
    #[new]
    fn new(ca_cert: String, client_cert: String, client_key: String) -> Self {
        TlsConfig(savant_core::eval_resolvers::TlsConfig {
            ca_cert,
            client_cert,
            client_key,
        })
    }
}

/// Registers the Etcd resolver in the system runtime.
///
/// Parameters
/// ----------
/// hosts: List[str]
///   The list of hosts to connect to the Etcd server.
///   Default is ["127.0.0.1:2379"].
/// credentials: Optional[EtcdCredentials]
///   The username and password to connect to the Etcd server.
///   Default is None.
/// tls_config: Optional[TlsConfig]
///   The TLS configuration to connect to the Etcd server.
///   Default is None.
/// watch_path: str
///   The path to watch for changes in the Etcd server.
///   Default is "savant".
/// connect_timeout: int
///   The timeout to connect to the Etcd server. In seconds.
///   Default is 5 seconds.
/// watch_path_wait_timeout: int
///   The timeout to wait for changes in the Etcd server. In seconds.
///   Default is 5 seconds.
///
#[pyfunction]
#[pyo3(signature = (hosts = vec!["127.0.0.1:2379".to_string()], credentials = None, tls_config = None, watch_path = "savant", connect_timeout = 5, watch_path_wait_timeout = 5))]
pub fn register_etcd_resolver(
    hosts: Vec<String>,
    credentials: Option<EtcdCredentials>,
    tls_config: Option<TlsConfig>,
    watch_path: &str,
    connect_timeout: u64,
    watch_path_wait_timeout: u64,
) -> PyResult<()> {
    let hosts_ref = hosts.iter().map(|s| s.as_str()).collect::<Vec<_>>();
    let creds = credentials.as_ref().map(|c| c.0.clone());
    let tls = tls_config.as_ref().map(|c| c.0.clone());
    savant_core::eval_resolvers::register_etcd_resolver(
        &hosts_ref,
        &creds,
        &tls,
        watch_path,
        connect_timeout,
        watch_path_wait_timeout,
    )
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Registers the Env resolver in the system runtime.
///
#[pyfunction]
pub fn register_env_resolver() {
    savant_core::eval_resolvers::register_env_resolver();
}

/// Registers the Utility resolver in the system runtime.
///
#[pyfunction]
pub fn register_utility_resolver() {
    savant_core::eval_resolvers::register_utility_resolver();
}

/// Registers the Config resolver in the system runtime.
///
/// Parameters
/// ----------
/// symbols: Dict[str, str]
///   The symbols to register in the Config resolver.
///
#[pyfunction]
pub fn register_config_resolver(symbols: HashMap<String, String>) {
    let symbols = symbols.into_iter().collect();
    savant_core::eval_resolvers::register_config_resolver(symbols);
}

/// Updates the Config resolver in the system runtime.
///
/// Parameters
/// ----------
/// symbols: Dict[str, str]
///   The symbols to update in the Config resolver.
///
#[pyfunction]
pub fn update_config_resolver(symbols: HashMap<String, String>) {
    let symbols = symbols.into_iter().collect();
    savant_core::eval_resolvers::update_config_resolver(symbols);
}

/// Unregisters the resolver from the system runtime.
///
/// Parameters
/// ----------
/// name: str
///   The name of the resolver to unregister.
///
#[pyfunction]
pub fn unregister_resolver(name: &str) {
    savant_core::eval_resolvers::unregister_resolver(name);
}
