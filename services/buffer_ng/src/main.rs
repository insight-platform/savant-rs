use anyhow::{Context, Result};
use log::info;
use pyo3::{
    types::{PyAnyMethods, PyDict, PyList, PyListMethods, PyModule},
    Py, PyAny, PyResult, Python,
};
use savant_core::{metrics::set_extra_labels, webserver::init_webserver};
use savant_core_py::webserver::set_status_running;
use savant_rs::LogLevel;
use std::env::args;

use buffer_ng::configuration::{InvocationContext, ServiceConfiguration};

const DEFAULT_CONFIGURATION_PATH: &str = "assets/configuration.json";

fn main() -> Result<()> {
    let mut python_handler: Option<Py<PyAny>> = None;
    ctrlc::set_handler(move || {
        info!("Ctrl+C received, shutting down...");
        std::process::exit(0);
    })?;

    savant_rs::init_logs(LogLevel::Info)?;

    info!("┌───────────────────────────────────────────────────────┐");
    info!("│               Savant Buffer NG Service                │");
    info!("│ This program is licensed under the APACHE 2.0 license │");
    info!("│      For more information, see the LICENSE file       │");
    info!("│            (c) 2025 BwSoft Management, LLC            │");
    info!("└───────────────────────────────────────────────────────┘");
    // python version
    Python::attach(|py| {
        let version = py.version();
        info!("Python version: {}", version);
    });
    // savant-rs version
    let savant_rs_version = savant_core::version();
    info!("Savant-rs library version: {}", savant_rs_version);

    let project_dir = env!("CARGO_MANIFEST_DIR");
    let default_config_path = format!("{}/{}", project_dir, DEFAULT_CONFIGURATION_PATH);
    let conf_arg = args().nth(1).unwrap_or(default_config_path);
    info!("Configuration: {}", conf_arg);
    let conf = ServiceConfiguration::new(&conf_arg)?;
    log::debug!("Configuration: {:?}", conf);

    // python handler init
    let py_handler_init_opt = conf.common.message_handler_init.as_ref();
    let (ingress_ph_opt, egress_ph_opt) = if let Some(py_handler_init) = py_handler_init_opt {
        let module_root = py_handler_init.python_root.as_str();
        let module_name = py_handler_init.module_name.as_str();
        let function_name = py_handler_init.function_name.as_str();
        let args = py_handler_init.args.as_ref();

        let json_str = if let Some(args) = args {
            serde_json::to_string(args)?
        } else {
            "null".to_string()
        };

        let invocation: PyResult<Py<PyAny>> = Python::attach(|py| {
            let json_module = PyModule::import(py, "json")?;
            let json_loads = json_module.getattr("loads")?;
            let py_data = json_loads.call1((&json_str,))?;

            let module = PyModule::new(py, "savant_rs")?;
            savant_rs::init_all(py, &module)?;
            let sys = PyModule::import(py, "sys")?;
            let sys_modules_bind = sys.getattr("modules")?;
            let sys_modules = sys_modules_bind.downcast::<PyDict>()?;
            sys_modules.set_item("savant_rs", module)?;
            let path_bind = sys.getattr("path")?;
            let path = path_bind.downcast::<PyList>()?;
            path.insert(0, module_root)?;
            let m = Python::import(py, module_name)?;
            let f = m.getattr(function_name)?.unbind();
            let res = f.call1(py, (py_data,))?;
            Ok(res)
        });

        let res = invocation?;
        Python::attach(|py| {
            if !res.is_none(py) {
                python_handler = Some(res);
            }
        });

        if python_handler.is_none() {
            log::info!(
                "Init module {} function {} returned None, working without python handler",
                module_name,
                function_name
            );
        } else {
            log::info!(
                "Init module {} function {} returned non-None, working with python handler, invocation context: {:?}",
                module_name,
                function_name,
                py_handler_init.invocation_context
            );
        }

        if let Some(h) = python_handler {
            match py_handler_init.invocation_context {
                InvocationContext::AfterReceive => (Some(h), None),
                InvocationContext::BeforeSend => (None, Some(h)),
            }
        } else {
            (None, None)
        }
    } else {
        (None, None)
    };

    init_webserver(conf.common.telemetry.port)?;
    info!(
        "Webserver initialized, port: {}",
        conf.common.telemetry.port
    );
    set_status_running()?;
    info!("Buffer NG status is set to running");
    if let Some(ref extra_labels) = conf.common.telemetry.metrics_extra_labels {
        let extra_labels = serde_json::from_value(extra_labels.clone())
            .with_context(|| format!("Failed to parse metrics extra labels: {:?}", extra_labels))?;
        info!("Metrics extra labels: {:?}", extra_labels);
        set_extra_labels(extra_labels);
    }

    buffer_ng::run_service_loop(&conf, ingress_ph_opt, egress_ph_opt, None)
}
