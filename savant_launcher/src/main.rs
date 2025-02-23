use clap::Parser;
use log::info;
use pyo3::prelude::*;
use pyo3::types::PyList;
use savant_rs::*;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[clap(short, long)]
    module_name: String,
    #[clap(short, long)]
    function_name: String,
    #[clap(short, long)]
    python_root: Option<String>,
    #[arg(last = true)]
    slop: Vec<String>,
}

fn main() -> anyhow::Result<()> {
    init_logs()?;
    let cli = Cli::parse();
    let module_root = cli.python_root.unwrap_or_else(|| ".".to_string());
    // check exists and is a directory
    let metadata = std::fs::metadata(&module_root)?;
    if !metadata.is_dir() {
        return Err(anyhow::anyhow!("{} is not a directory", module_root));
    }
    let module_name = cli.module_name;
    let function_name = cli.function_name;

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

    info!("GST_PLUGIN_PATH={}", std::env::var("GST_PLUGIN_PATH")?);

    let invocation: PyResult<()> = Python::with_gil(|py| {
        let module = PyModule::new(py, "savant_rs")?;
        init_all(py, &module)?;
        // add the current directory to the Python module load path
        let sys = PyModule::import(py, "sys")?;
        let path_bind = sys.as_ref().getattr("path")?;
        let path = path_bind.downcast::<PyList>()?;
        path.insert(0, module_root.as_str())?;
        let m = Python::import(py, module_name)?;
        let f = m.getattr(function_name.as_str())?.unbind();
        f.call0(py)?;
        Ok(())
    });
    invocation?;

    Ok(())
}
