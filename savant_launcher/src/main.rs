use pyo3::prelude::*;
use pyo3::types::PyList;
use savant_rs::*;

fn main() -> anyhow::Result<()> {
    init_logs();
    // get bootstrap directory is where the executable is located
    let gst_dir = std::env::current_exe()?
        .parent()
        .ok_or_else(|| anyhow::anyhow!("could not get parent directory of executable"))?
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("could not convert path to string"))?
        .to_string();

    let current_gst_plugin_path = std::env::var("GST_PLUGIN_PATH");
    unsafe {
        if let Ok(current_gst_plugin_path) = current_gst_plugin_path {
            std::env::set_var(
                "GST_PLUGIN_PATH",
                format!("{}:{}", current_gst_plugin_path, gst_dir),
            );
        } else {
            std::env::set_var("GST_PLUGIN_PATH", gst_dir);
        }
    }

    log::info!("GST_PLUGIN_PATH={}", std::env::var("GST_PLUGIN_PATH")?);

    // bootstrap directory is either "." or argv
    let bootstrap_module = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "./entrypoint.py".to_string());

    let bootstrap_dir = std::path::Path::new(&bootstrap_module)
        .parent()
        .ok_or_else(|| anyhow::anyhow!("could not get parent directory of bootstrap module"))?
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("could not convert path to string"))?
        .to_string();

    let bootstrap_module = std::path::Path::new(&bootstrap_module)
        .file_stem()
        .ok_or_else(|| anyhow::anyhow!("could not get file stem of bootstrap module"))?
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("could not convert path to string"))?
        .to_string();

    Python::with_gil(|py| {
        let module = PyModule::new(py, "savant_rs").map_err(|e| anyhow::anyhow!("{}", e))?;
        init_all(py, &module).map_err(|e| anyhow::anyhow!("{}", e))?;
        // add the current directory to the Python module load path
        let sys = PyModule::import(py, "sys")?;
        let path_bind = sys.as_ref().getattr("path")?;
        let path = path_bind
            .downcast::<PyList>()
            .map_err(|_| anyhow::anyhow!("sys.path is not a list"))?;
        path.insert(0, bootstrap_dir.as_str())?;
        Python::import(py, bootstrap_module)?;
        Ok(())
    })
}
