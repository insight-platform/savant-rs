mod configuration;

use anyhow::{anyhow, Result};
use configuration::ServiceConfiguration;
use log::{debug, info};
use pyo3::{
    types::{PyAnyMethods, PyBool, PyDict, PyList, PyListMethods, PyModule},
    PyResult, Python,
};

use savant_core_py::REGISTERED_HANDLERS;
use savant_rs::LogLevel;
use std::env::args;

use crate::configuration::{
    CREATE_STREAMS_REQUEST_LABEL, FRAME_EVENT_LABEL, PROBE_EVENT_LABEL, STOP_STREAMS_REQUEST_LABEL,
    STREAM_TERMINATION_EVENT_LABEL,
};

fn main() -> Result<()> {
    ctrlc::set_handler(move || {
        info!("Ctrl+C received, shutting down...");
        std::process::exit(0);
    })?;

    savant_rs::init_logs(LogLevel::Info)?;

    info!("┌───────────────────────────────────────────────────────┐");
    info!("│           Savant FFmpeg Source Service                │");
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

    let conf_arg = args()
        .nth(1)
        .ok_or_else(|| anyhow!("missing configuration argument"))?;

    info!("Configuration: {}", conf_arg);
    let conf = ServiceConfiguration::new(&conf_arg)?;
    debug!("Configuration: {:?}", conf);

    let module_root = conf.event_handlers.python_root.as_str();
    let module_name = conf.event_handlers.module_name.as_str();
    let function_name = conf.event_handlers.function_name.as_str();
    let args = conf.event_handlers.args.as_ref();

    let json_str = if let Some(args) = args {
        serde_json::to_string(args)?
    } else {
        "null".to_string()
    };

    let invocation: PyResult<bool> = Python::attach(|py| {
        let json_module = PyModule::import(py, "json")?;
        let json_loads = json_module.getattr("loads")?;
        let py_data = json_loads.call1((&json_str,))?;

        let module = PyModule::new(py, "savant_rs")?;
        savant_rs::init_all(py, &module)?;
        // add the current directory to the Python module load path
        let sys = PyModule::import(py, "sys")?;
        let sys_modules_bind = sys.getattr("modules")?;
        let sys_modules = sys_modules_bind.downcast::<PyDict>()?;
        sys_modules.set_item("savant_rs", module)?;
        let path_bind = sys.getattr("path")?;
        let path = path_bind.downcast::<PyList>()?;
        path.insert(0, module_root)?;
        let m = Python::import(py, module_name)?;
        let f = m.getattr(function_name)?.unbind();
        let res = f.call1(py, (py_data,))?.into_bound(py);
        res.downcast::<PyBool>()?.extract()
    });
    let res = invocation?;
    if res {
        let handlers_bind = REGISTERED_HANDLERS.read();
        if conf.event_handlers.callbacks.frame_event {
            if !handlers_bind.contains_key(FRAME_EVENT_LABEL) {
                return Err(anyhow!(
                    "Frame event handler ({:?}) is required but is not registered",
                    FRAME_EVENT_LABEL
                ));
            }
        }
        if conf.event_handlers.callbacks.probe_event {
            if !handlers_bind.contains_key(PROBE_EVENT_LABEL) {
                return Err(anyhow!(
                    "Probe event handler ({:?}) is required but is not registered",
                    PROBE_EVENT_LABEL
                ));
            }
        }
        if conf.event_handlers.callbacks.stream_termination_event {
            if !handlers_bind.contains_key(STREAM_TERMINATION_EVENT_LABEL) {
                return Err(anyhow!(
                    "Stream termination event handler ({:?}) is required but is not registered",
                    STREAM_TERMINATION_EVENT_LABEL
                ));
            }
        }
        if !handlers_bind.contains_key(CREATE_STREAMS_REQUEST_LABEL) {
            return Err(anyhow!(
                "Create streams request handler ({:?}) is required but is not registered",
                CREATE_STREAMS_REQUEST_LABEL
            ));
        }

        if !handlers_bind.contains_key(STOP_STREAMS_REQUEST_LABEL) {
            return Err(anyhow!(
                "Stop streams request handler ({:?}) is required but is not registered",
                STOP_STREAMS_REQUEST_LABEL
            ));
        }

        info!("FFmpeg source service started successfully");
    } else {
        return Err(anyhow!(
            "FFmpeg source service failed to start. The initialized function {} in module {} returned false",
            function_name,
            module_name
        ));
    }

    Ok(())
}
