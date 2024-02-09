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

/// Registers the Etcd resolver in the system runtime.
///
/// Params
/// ------
/// hosts: List[str]
///   The list of hosts to connect to the Etcd server.
///   Default is ["127.0.0.1:2379"].
/// credentials: Optional[Tuple[str, str]]
///   The username and password to connect to the Etcd server.
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
#[pyo3(signature = (hosts = vec!["127.0.0.1:2379".to_string()], credentials = None, watch_path = "savant", connect_timeout = 5, watch_path_wait_timeout = 5))]
pub fn register_etcd_resolver(
    hosts: Vec<String>,
    credentials: Option<(String, String)>,
    watch_path: &str,
    connect_timeout: u64,
    watch_path_wait_timeout: u64,
) -> PyResult<()> {
    let hosts_ref = hosts.iter().map(|s| s.as_str()).collect::<Vec<_>>();
    let creds = credentials.as_ref().map(|(a, b)| (a.as_str(), b.as_str()));
    savant_core::eval_resolvers::register_etcd_resolver(
        &hosts_ref,
        &creds,
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
/// Params
/// ------
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
/// Params
/// ------
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
/// Params
/// ------
/// name: str
///   The name of the resolver to unregister.
///
#[pyfunction]
pub fn unregister_resolver(name: &str) {
    savant_core::eval_resolvers::unregister_resolver(name);
}
