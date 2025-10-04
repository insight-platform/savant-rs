mod configuration;
mod message_handler;
mod rocksdb;
use ::rocksdb::Options;
use anyhow::Result;
use log::{debug, info};
use parking_lot::Mutex;
use pyo3::{
    types::{PyAnyMethods, PyDict, PyList, PyListMethods, PyModule},
    Py, PyAny, PyResult, Python,
};
use savant_core::transport::zeromq::{NonBlockingReader, NonBlockingWriter, ReaderResult};
use savant_rs::LogLevel;
use savant_services_common::topic_to_string;
use std::{env::args, sync::Arc};

use crate::{
    configuration::{InvocationContext, ServiceConfiguration},
    message_handler::{MessageHandler, MessageWriter},
    rocksdb::PersistentQueueWithCapacity,
};

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
    debug!("Configuration: {:?}", conf);

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
            let res = f.call1(py, (py_data,))?;
            Ok(res)
        });

        let res = invocation?;
        // if not none, set the python_handler
        Python::attach(|py| {
            if !res.is_none(py) {
                python_handler = Some(res);
            }
        });

        if python_handler.is_none() {
            // working without python handler
            log::info!(
                "Init module {} function {} returned None, working without python handler",
                module_name,
                function_name
            );
        } else {
            // working with python handler
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

    let reader = NonBlockingReader::try_from(&conf.ingress.socket)?;
    let writer = NonBlockingWriter::try_from(&conf.egress.socket)?;
    let db_opts = Options::default();
    if conf.common.buffer.reset_on_start {
        PersistentQueueWithCapacity::remove_db(&conf.common.buffer.path)?;
    }
    let queue = PersistentQueueWithCapacity::new(
        &conf.common.buffer.path,
        conf.common.buffer.max_length,
        conf.common.buffer.full_threshold_percentage,
        db_opts,
    )?;
    log::info!(
        "Buffer initialized, path: {}, max length: {}, \
        full threshold: {}, reset on start: {}, \
        current length: {}, current disk size: {}",
        conf.common.buffer.path,
        conf.common.buffer.max_length,
        conf.common.buffer.full_threshold_percentage,
        conf.common.buffer.reset_on_start,
        queue.len(),
        queue.disk_size()?
    );

    let queue = Arc::new(Mutex::new(queue));
    let mut message_writer = MessageWriter::new(queue.clone(), ingress_ph_opt);
    let mut message_handler =
        MessageHandler::new(queue, writer, conf.common.idle_sleep, egress_ph_opt);

    std::thread::spawn(move || loop {
        let res = message_handler.process_stored_message();
        if let Err(e) = res {
            log::warn!(
                target: "buffer_ng::message_handler",
                "Failed to process message delivery: {:?}",
                e
            );
        }
    });

    loop {
        let message = reader.receive()?;
        match message {
            ReaderResult::Message {
                message,
                topic,
                routing_id: _,
                data,
            } => message_writer.push(topic_to_string(&topic), *message, data)?,
            ReaderResult::Timeout => {
                debug!(
                    target: "buffer_ng::ingress",
                    "Timeout receiving message, waiting for next message."
                );
            }
            ReaderResult::PrefixMismatch { topic, routing_id } => {
                log::warn!(
                    target: "buffer_ng::ingress",
                    "Received message with mismatched prefix: topic: {:?}, routing_id: {:?}",
                    topic_to_string(&topic),
                    topic_to_string(routing_id.as_ref().unwrap_or(&Vec::new()))
                );
            }
            ReaderResult::RoutingIdMismatch { topic, routing_id } => {
                log::warn!(
                    target: "buffer_ng::ingress",
                    "Received message with mismatched routing_id: topic: {:?}, routing_id: {:?}",
                    topic_to_string(&topic),
                    topic_to_string(routing_id.as_ref().unwrap_or(&Vec::new()))
                );
            }
            ReaderResult::TooShort(m) => {
                log::warn!(
                    target: "buffer_ng::ingress",
                    "Received message that was too short: {:?}",
                    m
                );
            }
            ReaderResult::MessageVersionMismatch {
                topic,
                routing_id,
                sender_version,
                expected_version,
            } => {
                log::warn!(
                    target: "buffer_ng::ingress",
                    "Received message with mismatched version: topic: {:?}, routing_id: {:?}, sender_version: {:?}, expected_version: {:?}",
                    topic_to_string(&topic),
                    topic_to_string(routing_id.as_ref().unwrap_or(&Vec::new())),
                    sender_version,
                    expected_version
                );
            }
            ReaderResult::Blacklisted(items) => {
                log::warn!(
                    target: "buffer_ng::ingress",
                    "Received blacklisted message: {:?}",
                    items
                );
            }
        }
    }
}
