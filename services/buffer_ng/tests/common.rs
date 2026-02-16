#![allow(dead_code)]

use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::Result;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

use buffer_ng::configuration::{
    BufferConfiguration, CommonConfiguration, EgressConfiguration, IngressConfiguration,
    ServiceConfiguration, TelemetryConfiguration,
};
use savant_core::message::Message;
use savant_core::primitives::attribute_value::{AttributeValue, AttributeValueVariant};
use savant_core::primitives::WithAttributes;
use savant_core::test::gen_frame;
use savant_core::transport::zeromq::{
    NonBlockingReader, NonBlockingWriter, ReaderConfig, WriterConfig,
};
use savant_services_common::job_writer::SinkConfiguration;
use savant_services_common::source::{SourceConfiguration, SourceOptions, TopicPrefixSpec};

// ══════════════════════════════════════════════════════════════════════════
// PyO3 handler classes
// ══════════════════════════════════════════════════════════════════════════

/// Passthrough handler: returns (topic, message) unchanged.
#[pyclass]
struct PassthroughHandler;

#[pymethods]
impl PassthroughHandler {
    #[new]
    fn new() -> Self {
        Self
    }
    fn __call__(
        &self,
        topic: &str,
        message: Bound<'_, PyAny>,
    ) -> (String, Py<PyAny>) {
        (topic.to_string(), message.unbind())
    }
}

/// Handler that stamps an attribute on every video frame it processes.
/// Used to verify that the handler was actually invoked.
#[pyclass]
struct StampingHandler {
    namespace: String,
    label: String,
}

#[pymethods]
impl StampingHandler {
    #[new]
    fn new(namespace: String, label: String) -> Self {
        Self { namespace, label }
    }
    fn __call__(
        &self,
        topic: &str,
        message: Bound<'_, PyAny>,
    ) -> PyResult<(String, Py<PyAny>)> {
        let is_vf: bool = message.call_method0("is_video_frame")?.extract()?;
        if is_vf {
            let mut frame: savant_core_py::primitives::frame::VideoFrame =
                message.call_method0("as_video_frame")?.extract()?;
            frame.0.set_persistent_attribute(
                &self.namespace,
                &self.label,
                &None,
                false,
                vec![AttributeValue::boolean(true, None)],
            );
        }
        Ok((topic.to_string(), message.unbind()))
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Python initialisation
// ══════════════════════════════════════════════════════════════════════════

static PYTHON_INIT: std::sync::Once = std::sync::Once::new();

pub fn init_python() {
    PYTHON_INIT.call_once(|| {
        Python::attach(|py| -> PyResult<()> {
            let module = PyModule::new(py, "savant_rs")?;
            savant_rs::init_all(py, &module)?;
            let sys = PyModule::import(py, "sys")?;
            let sys_modules_bind = sys.getattr("modules")?;
            let sys_modules = sys_modules_bind.downcast::<PyDict>()?;
            sys_modules.set_item("savant_rs", module)?;
            Ok(())
        })
        .expect("Failed to initialise Python");
    });
}

// ══════════════════════════════════════════════════════════════════════════
// Handler factories
// ══════════════════════════════════════════════════════════════════════════

pub fn make_passthrough_handler() -> Py<PyAny> {
    Python::attach(|py| {
        Py::new(py, PassthroughHandler::new())
            .unwrap()
            .into_any()
    })
}

pub fn make_stamping_handler(namespace: &str, label: &str) -> Py<PyAny> {
    Python::attach(|py| {
        Py::new(
            py,
            StampingHandler::new(namespace.to_string(), label.to_string()),
        )
        .unwrap()
        .into_any()
    })
}

// ══════════════════════════════════════════════════════════════════════════
// IPC address helper
// ══════════════════════════════════════════════════════════════════════════

pub fn ipc_addr(test: &str, role: &str) -> String {
    format!(
        "ipc:///tmp/buffer_ng_{}_{}_{}",
        test,
        role,
        std::process::id()
    )
}

// ══════════════════════════════════════════════════════════════════════════
// Configuration builders
// ══════════════════════════════════════════════════════════════════════════

pub fn make_service_conf(
    ingress_ipc: &str,
    egress_ipc: &str,
    buffer_path: &str,
) -> ServiceConfiguration {
    ServiceConfiguration {
        ingress: IngressConfiguration {
            socket: SourceConfiguration {
                url: format!("rep+bind:{}", ingress_ipc),
                options: Some(SourceOptions {
                    receive_timeout: Duration::from_millis(100),
                    receive_hwm: 1000,
                    topic_prefix_spec: TopicPrefixSpec::None,
                    source_cache_size: 100,
                    fix_ipc_permissions: None,
                    inflight_ops: 100,
                }),
            },
        },
        egress: EgressConfiguration {
            socket: SinkConfiguration {
                url: format!("dealer+connect:{}", egress_ipc),
                options: None,
            },
        },
        common: CommonConfiguration {
            message_handler_init: None,
            telemetry: TelemetryConfiguration {
                port: 0,
                stats_log_interval: Duration::from_secs(600),
                metrics_extra_labels: None,
            },
            buffer: BufferConfiguration {
                path: buffer_path.to_string(),
                max_length: 10_000,
                full_threshold_percentage: 90,
                reset_on_start: true,
            },
            idle_sleep: Duration::from_millis(1),
        },
    }
}

// ══════════════════════════════════════════════════════════════════════════
// ZMQ helpers
// ══════════════════════════════════════════════════════════════════════════

pub fn start_dest_reader(ipc: &str) -> Result<NonBlockingReader> {
    let conf = ReaderConfig::new()
        .url(&format!("router+bind:{}", ipc))?
        .with_receive_timeout(100)?
        .build()?;
    let mut reader = NonBlockingReader::new(&conf, 100)?;
    reader.start()?;
    thread::sleep(Duration::from_millis(200));
    Ok(reader)
}

pub fn start_source_writer(ipc: &str) -> Result<NonBlockingWriter> {
    let conf = WriterConfig::new()
        .url(&format!("req+connect:{}", ipc))?
        .with_send_timeout(1000)?
        .with_receive_timeout(1000)?
        .with_send_retries(3)?
        .with_receive_retries(3)?
        .build()?;
    let mut writer = NonBlockingWriter::new(&conf, 100)?;
    writer.start()?;
    thread::sleep(Duration::from_millis(200));
    Ok(writer)
}

pub fn start_service(
    conf: ServiceConfiguration,
    ingress_handler: Option<Py<PyAny>>,
    egress_handler: Option<Py<PyAny>>,
    shutdown: Arc<AtomicBool>,
) -> JoinHandle<Result<()>> {
    let handle = thread::spawn(move || {
        buffer_ng::run_service_loop(&conf, ingress_handler, egress_handler, Some(shutdown))
    });
    thread::sleep(Duration::from_millis(500));
    handle
}

// ══════════════════════════════════════════════════════════════════════════
// Test-step helpers
// ══════════════════════════════════════════════════════════════════════════

pub struct TestStep {
    pub message: Option<Message>,
    pub is_eos: bool,
}

impl TestStep {
    pub fn frame() -> Self {
        Self {
            message: Some(Message::video_frame(&gen_frame())),
            is_eos: false,
        }
    }

    pub fn eos() -> Self {
        Self {
            message: None,
            is_eos: true,
        }
    }

    pub fn send(&self, writer: &NonBlockingWriter) -> Result<()> {
        if self.is_eos {
            writer.send_eos("test")?.get()?;
        } else {
            writer
                .send_message("test", self.message.as_ref().unwrap(), &[])?
                .get()?;
        }
        Ok(())
    }
}

/// N frames followed by an EOS.
pub fn frame_eos_steps(num_frames: u32) -> impl Iterator<Item = TestStep> {
    (0..num_frames)
        .map(|_| TestStep::frame())
        .chain(std::iter::once(TestStep::eos()))
}

/// Wait (up to `deadline`) for the next message at `reader` accepted by
/// `accept`.  Returns `true` when the predicate matched before the deadline.
pub fn wait_for_dest<F>(reader: &NonBlockingReader, deadline: Instant, mut accept: F) -> bool
where
    F: FnMut(&Message) -> bool,
{
    while Instant::now() < deadline {
        match reader.try_receive() {
            Some(Ok(savant_core::transport::zeromq::ReaderResult::Message { message, .. })) => {
                if accept(&message) {
                    return true;
                }
            }
            _ => thread::sleep(Duration::from_millis(5)),
        }
    }
    false
}

/// Assert that a video frame carries `(namespace, label)` as a boolean-true
/// persistent attribute.
pub fn assert_has_stamp(frame: &savant_core::primitives::frame::VideoFrameProxy, ns: &str, label: &str) {
    let attr = frame
        .get_attribute(ns, label)
        .unwrap_or_else(|| panic!("Frame should have ({ns}, {label}) attribute"));
    match attr.values.as_ref()[0].get() {
        AttributeValueVariant::Boolean(v) => assert!(v, "({ns}, {label}) should be true"),
        other => panic!("({ns}, {label}) expected Boolean, got {:?}", other),
    }
}
