mod configuration;

use anyhow::{anyhow, Result};
use configuration::ServiceConfiguration;
use log::{debug, info};
use pyo3::{
    types::{PyAnyMethods, PyBool, PyDict, PyList, PyListMethods, PyModule},
    PyResult, Python,
};
use std::{env::args, sync::Arc};

fn main() -> Result<()> {
    savant_rs::init_logs()?;

    info!("┌───────────────────────────────────────────────────────┐");
    info!("│                Savant Router Service                  │");
    info!("│ This program is licensed under the APACHE 2.0 license │");
    info!("│      For more information, see the LICENSE file       │");
    info!("│            (c) 2025 BwSoft Management, LLC            │");
    info!("└───────────────────────────────────────────────────────┘");

    let conf_arg = args()
        .nth(1)
        .ok_or_else(|| anyhow!("missing configuration argument"))?;
    info!("Configuration: {}", conf_arg);
    let conf = Arc::new(ServiceConfiguration::new(&conf_arg)?);
    debug!("Configuration: {:?}", conf);

    let module_root = conf.common.init.as_ref().unwrap().python_root.as_str();
    let module_name = conf.common.init.as_ref().unwrap().module_name.as_str();
    let function_name = conf.common.init.as_ref().unwrap().function_name.as_str();
    let args = conf.common.init.as_ref().unwrap().args.as_ref().unwrap();
    let _args_as_json = serde_json::to_string(args)?;

    let invocation: PyResult<bool> = Python::with_gil(|py| {
        let module = PyModule::new(py, "savant_rs")?;
        savant_rs::init_all(py, &module)?;
        // add the current directory to the Python module load path
        let sys = PyModule::import(py, "sys")?;
        let sys_modules_bind = sys.getattr("modules")?;
        let sys_modules = sys_modules_bind.downcast::<PyDict>()?;
        sys_modules.set_item("savant_rs", module)?;
        let path_bind = sys.as_ref().getattr("path")?;
        let path = path_bind.downcast::<PyList>()?;
        path.insert(0, module_root)?;
        let m = Python::import(py, module_name)?;
        let f = m.getattr(function_name)?.unbind();
        let res = f.call0(py)?.into_bound(py);
        Ok(res.downcast::<PyBool>()?.extract()?)
    });
    let res = invocation?;
    if res {
        info!("Router service started successfully");
    } else {
        return Err(anyhow!(
            "Router service failed to start. The initialized function {} in module {} returned false",
            function_name,
            module_name
        ));
    }

    Ok(())
}
