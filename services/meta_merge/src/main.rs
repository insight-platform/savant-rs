use anyhow::{anyhow, Result};
use meta_merge::configuration::ServiceConfiguration;
use meta_merge::egress::{egress::Egress, processor::EgressProcessor};
use meta_merge::ingress::Ingress;
use log::{debug, info};
use pyo3::{
    types::{PyAnyMethods, PyBool, PyDict, PyList, PyListMethods, PyModule},
    PyResult, Python,
};
use savant_core::transport::zeromq::NonBlockingWriter;
use savant_core_py::logging::LogLevel;
use std::env::args;

fn main() -> Result<()> {
    ctrlc::set_handler(move || {
        info!("Ctrl+C received, shutting down...");
        std::process::exit(0);
    })?;

    savant_rs::init_logs(LogLevel::Info)?;

    info!("┌───────────────────────────────────────────────────────┐");
    info!("│               Savant Meta Merge Service               │");
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

    let module_root = conf.common.init.as_ref().unwrap().python_root.as_str();
    let module_name = conf.common.init.as_ref().unwrap().module_name.as_str();
    let function_name = conf.common.init.as_ref().unwrap().function_name.as_str();
    let args = conf.common.init.as_ref().unwrap().args.as_ref();

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
        info!("Meta Merge service started successfully");
    } else {
        return Err(anyhow!(
            "Meta Merge service failed to start. The initialized function {} in module {} returned false",
            function_name,
            module_name
        ));
    }

    let mut ingress = Ingress::new(&conf)?;

    // Create egress buffer and writer
    let buffer = Egress::new(conf.common.queue.max_duration);
    let writer = NonBlockingWriter::try_from(&conf.egress.socket)?;
    let mut processor = EgressProcessor::new(buffer, conf.common.callbacks.clone(), writer);

    loop {
        let messages = ingress.get()?;
        if messages.is_empty() {
            // Even when idle, poll for expired heads
            processor.send_ready()?;
            std::thread::sleep(conf.common.idle_sleep);
            debug!(
                "No messages received, sleeping for {:?}",
                conf.common.idle_sleep
            );
            continue;
        }
        for ingress_message in messages {
            let topic = &ingress_message.topic;
            let message = ingress_message.message;
            let data = ingress_message.data;
            let ingress_name = &ingress_message.ingress_name;

            if message.is_video_frame() {
                let frame_proxy = message.as_video_frame().unwrap();
                let frame =
                    savant_core_py::primitives::frame::VideoFrame(frame_proxy);
                let labels = message.get_labels();
                processor.process_frame(ingress_name, topic, frame, data, labels)?;
            } else if message.is_end_of_stream() {
                let eos = message.as_end_of_stream().unwrap();
                let source_id = eos.get_source_id().to_string();
                let labels = message.get_labels();
                processor.process_eos(source_id, data, labels)?;
            }
        }

        // After processing all messages, send any ready heads
        processor.send_ready()?;
    }
}
