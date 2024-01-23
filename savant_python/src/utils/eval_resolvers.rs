use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::collections::HashMap;

#[pyfunction]
pub fn utility_resolver_name() -> &'static str {
    savant_core::eval_resolvers::utility_resolver_name()
}

#[pyfunction]
pub fn env_resolver_name() -> &'static str {
    savant_core::eval_resolvers::env_resolver_name()
}

#[pyfunction]
pub fn config_resolver_name() -> &'static str {
    savant_core::eval_resolvers::config_resolver_name()
}

#[pyfunction]
pub fn etcd_resolver_name() -> &'static str {
    savant_core::eval_resolvers::etcd_resolver_name()
}

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

#[pyfunction]
pub fn register_env_resolver() {
    savant_core::eval_resolvers::register_env_resolver();
}

#[pyfunction]
pub fn register_utility_resolver() {
    savant_core::eval_resolvers::register_utility_resolver();
}

#[pyfunction]
pub fn register_config_resolver(symbols: HashMap<String, String>) {
    let symbols = symbols.into_iter().collect();
    savant_core::eval_resolvers::register_config_resolver(symbols);
}

#[pyfunction]
pub fn update_config_resolver(symbols: HashMap<String, String>) {
    let symbols = symbols.into_iter().collect();
    savant_core::eval_resolvers::update_config_resolver(symbols);
}

#[pyfunction]
pub fn unregister_resolver(name: &str) {
    savant_core::eval_resolvers::unregister_resolver(name);
}
