#![allow(dead_code)]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::Result;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

use meta_merge::configuration::{
    CallbacksConfiguration, EosPolicy, IngressConfiguration, ServiceConfiguration,
};
use savant_core::message::Message;
use savant_core::primitives::userdata::UserData;
use savant_core::primitives::WithAttributes;
use savant_core::test::gen_frame;
use savant_core::transport::zeromq::{
    NonBlockingReader, NonBlockingWriter, ReaderConfig, WriterConfig,
};
use savant_core_py::REGISTERED_HANDLERS;
use savant_services_common::source::{SourceConfiguration, SourceOptions, TopicPrefixSpec};

// ══════════════════════════════════════════════════════════════════════════
// Shared statics
// ══════════════════════════════════════════════════════════════════════════

pub static UNSUPPORTED_HANDLER_CALLED: AtomicBool = AtomicBool::new(false);
pub static LATE_ARRIVAL_CALLED: AtomicBool = AtomicBool::new(false);

// ══════════════════════════════════════════════════════════════════════════
// PyO3 handler classes
// ══════════════════════════════════════════════════════════════════════════

/// Merge handler that immediately marks the frame as ready.
#[pyclass]
struct AlwaysReadyMergeHandler;

#[pymethods]
impl AlwaysReadyMergeHandler {
    #[new]
    fn new() -> Self {
        Self
    }
    fn __call__(
        &self,
        _ingress_name: &str,
        _topic: &str,
        _current_state: Bound<'_, PyAny>,
        _incoming_state: Option<Bound<'_, PyAny>>,
    ) -> bool {
        true
    }
}

/// Merge handler that never marks the frame as ready (waits for expiry).
#[pyclass]
struct NeverReadyMergeHandler;

#[pymethods]
impl NeverReadyMergeHandler {
    #[new]
    fn new() -> Self {
        Self
    }
    fn __call__(
        &self,
        _ingress_name: &str,
        _topic: &str,
        _current_state: Bound<'_, PyAny>,
        _incoming_state: Option<Bound<'_, PyAny>>,
    ) -> bool {
        false
    }
}

/// Merge handler for two-stream scenarios.
///
/// - First arrival (`incoming_state` is `None`): sets `(merge, merge_count)`
///   to `1` (integer) and `(merge, first_ingress)` to the ingress name
///   (string).  Returns `false`.
/// - Second arrival (`incoming_state` is `Some`): sets `(merge, merge_count)`
///   to `2` and `(merge, second_ingress)` to the ingress name.  Returns
///   `true` (ready).
#[pyclass]
struct TwoStreamMergeHandler;

#[pymethods]
impl TwoStreamMergeHandler {
    #[new]
    fn new() -> Self {
        Self
    }
    fn __call__(
        &self,
        ingress_name: &str,
        _topic: &str,
        current_state: Bound<'_, PyAny>,
        incoming_state: Option<Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        use savant_core::primitives::attribute_value::AttributeValue;

        let mut vf: savant_core_py::primitives::frame::VideoFrame =
            current_state.getattr("video_frame")?.extract()?;

        if incoming_state.is_some() {
            vf.0.set_persistent_attribute(
                "merge",
                "merge_count",
                &None,
                false,
                vec![AttributeValue::integer(2, None)],
            );
            vf.0.set_persistent_attribute(
                "merge",
                "second_ingress",
                &None,
                false,
                vec![AttributeValue::string(ingress_name, None)],
            );
            Ok(true)
        } else {
            vf.0.set_persistent_attribute(
                "merge",
                "merge_count",
                &None,
                false,
                vec![AttributeValue::integer(1, None)],
            );
            vf.0.set_persistent_attribute(
                "merge",
                "first_ingress",
                &None,
                false,
                vec![AttributeValue::string(ingress_name, None)],
            );
            Ok(false)
        }
    }
}

/// Head-expired handler: sets `(merge, head_expired)` attribute, then wraps
/// the frame into a Message and returns it.
#[pyclass]
struct TestHeadExpiredHandler;

#[pymethods]
impl TestHeadExpiredHandler {
    #[new]
    fn new() -> Self {
        Self
    }
    fn __call__(&self, state: Bound<'_, PyAny>) -> PyResult<Option<Py<PyAny>>> {
        let py = state.py();
        let mut video_frame: savant_core_py::primitives::frame::VideoFrame =
            state.getattr("video_frame")?.extract()?;
        video_frame.0.set_persistent_attribute(
            "merge",
            "head_expired",
            &None,
            false,
            vec![savant_core::primitives::attribute_value::AttributeValue::boolean(true, None)],
        );
        let msg = savant_core_py::primitives::message::Message::video_frame(&video_frame);
        Ok(Some(msg.into_pyobject(py)?.unbind().into()))
    }
}

/// Head-ready handler: sets `(merge, head_ready)` attribute, then returns
/// the message.
#[pyclass]
struct TestHeadReadyHandler;

#[pymethods]
impl TestHeadReadyHandler {
    #[new]
    fn new() -> Self {
        Self
    }
    fn __call__(&self, state: Bound<'_, PyAny>) -> PyResult<Option<Py<PyAny>>> {
        let py = state.py();
        let mut video_frame: savant_core_py::primitives::frame::VideoFrame =
            state.getattr("video_frame")?.extract()?;
        video_frame.0.set_persistent_attribute(
            "merge",
            "head_ready",
            &None,
            false,
            vec![savant_core::primitives::attribute_value::AttributeValue::boolean(true, None)],
        );
        let msg = savant_core_py::primitives::message::Message::video_frame(&video_frame);
        Ok(Some(msg.into_pyobject(py)?.unbind().into()))
    }
}

/// Late-arrival handler: no-op.
#[pyclass]
struct TestLateArrivalHandler;

#[pymethods]
impl TestLateArrivalHandler {
    #[new]
    fn new() -> Self {
        Self
    }
    fn __call__(&self, _state: Bound<'_, PyAny>) -> PyResult<()> {
        Ok(())
    }
}

/// Send handler: sets `(merge, sent)` attribute on video frames, returns
/// `None` (use default topic).
#[pyclass]
struct TestSendHandler;

#[pymethods]
impl TestSendHandler {
    #[new]
    fn new() -> Self {
        Self
    }
    fn __call__(
        &self,
        message: Bound<'_, PyAny>,
        _message_state: Bound<'_, PyAny>,
        _data: Bound<'_, PyAny>,
        _labels: Bound<'_, PyAny>,
    ) -> PyResult<Option<String>> {
        let is_vf: bool = message.call_method0("is_video_frame")?.extract()?;
        if is_vf {
            let mut frame: savant_core_py::primitives::frame::VideoFrame =
                message.call_method0("as_video_frame")?.extract()?;
            frame.0.set_persistent_attribute(
                "merge",
                "sent",
                &None,
                false,
                vec![savant_core::primitives::attribute_value::AttributeValue::boolean(true, None)],
            );
        }
        Ok(None)
    }
}

/// Unsupported-message handler: sets a flag when invoked.
#[pyclass]
struct TestUnsupportedMessageHandler;

#[pymethods]
impl TestUnsupportedMessageHandler {
    #[new]
    fn new() -> Self {
        Self
    }
    fn __call__(
        &self,
        _ingress_name: &str,
        _topic: &str,
        _message: Bound<'_, PyAny>,
        _data: Bound<'_, PyAny>,
    ) -> PyResult<()> {
        UNSUPPORTED_HANDLER_CALLED.store(true, Ordering::SeqCst);
        Ok(())
    }
}

/// Head-ready handler that drops frames (returns None).
#[pyclass]
struct DropHeadReadyHandler;

#[pymethods]
impl DropHeadReadyHandler {
    #[new]
    fn new() -> Self {
        Self
    }
    fn __call__(&self, _state: Bound<'_, PyAny>) -> PyResult<Option<Py<PyAny>>> {
        Ok(None)
    }
}

/// Head-expired handler that drops frames (returns None).
#[pyclass]
struct DropHeadExpiredHandler;

#[pymethods]
impl DropHeadExpiredHandler {
    #[new]
    fn new() -> Self {
        Self
    }
    fn __call__(&self, _state: Bound<'_, PyAny>) -> PyResult<Option<Py<PyAny>>> {
        Ok(None)
    }
}

/// Send handler that overrides topic to "custom_topic".
#[pyclass]
struct TopicOverrideSendHandler;

#[pymethods]
impl TopicOverrideSendHandler {
    #[new]
    fn new() -> Self {
        Self
    }
    fn __call__(
        &self,
        _message: Bound<'_, PyAny>,
        _message_state: Bound<'_, PyAny>,
        _data: Bound<'_, PyAny>,
        _labels: Bound<'_, PyAny>,
    ) -> PyResult<Option<String>> {
        Ok(Some("custom_topic".into()))
    }
}

/// Late-arrival handler that records invocation in an AtomicBool.
#[pyclass]
struct TrackingLateArrivalHandler;

#[pymethods]
impl TrackingLateArrivalHandler {
    #[new]
    fn new() -> Self {
        Self
    }
    fn __call__(&self, _state: Bound<'_, PyAny>) -> PyResult<()> {
        LATE_ARRIVAL_CALLED.store(true, Ordering::SeqCst);
        Ok(())
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Shared helpers
// ══════════════════════════════════════════════════════════════════════════

/// One-time Python + savant_rs module initialisation.
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

pub fn register_all_handlers() {
    Python::attach(|py| {
        let always_ready = Py::new(py, AlwaysReadyMergeHandler::new()).unwrap();
        let never_ready = Py::new(py, NeverReadyMergeHandler::new()).unwrap();
        let two_stream = Py::new(py, TwoStreamMergeHandler::new()).unwrap();
        let head_expired = Py::new(py, TestHeadExpiredHandler::new()).unwrap();
        let head_ready = Py::new(py, TestHeadReadyHandler::new()).unwrap();
        let late_arrival = Py::new(py, TestLateArrivalHandler::new()).unwrap();
        let send = Py::new(py, TestSendHandler::new()).unwrap();
        let unsupported = Py::new(py, TestUnsupportedMessageHandler::new()).unwrap();
        let drop_head_ready = Py::new(py, DropHeadReadyHandler::new()).unwrap();
        let drop_head_expired = Py::new(py, DropHeadExpiredHandler::new()).unwrap();
        let topic_override_send = Py::new(py, TopicOverrideSendHandler::new()).unwrap();
        let tracking_late = Py::new(py, TrackingLateArrivalHandler::new()).unwrap();

        let mut h = REGISTERED_HANDLERS.write();
        h.insert("merge_always_ready".into(), always_ready.into_any());
        h.insert("merge_never_ready".into(), never_ready.into_any());
        h.insert("merge_two_stream".into(), two_stream.into_any());
        h.insert("head_expired_handler".into(), head_expired.into_any());
        h.insert("head_ready_handler".into(), head_ready.into_any());
        h.insert("late_arrival_handler".into(), late_arrival.into_any());
        h.insert("send_handler".into(), send.into_any());
        h.insert("unsupported_message_handler".into(), unsupported.into_any());
        h.insert("drop_head_ready_handler".into(), drop_head_ready.into_any());
        h.insert(
            "drop_head_expired_handler".into(),
            drop_head_expired.into_any(),
        );
        h.insert(
            "topic_override_send_handler".into(),
            topic_override_send.into_any(),
        );
        h.insert(
            "tracking_late_arrival_handler".into(),
            tracking_late.into_any(),
        );
    });
}

pub fn unregister_all_handlers() {
    REGISTERED_HANDLERS.write().clear();
}

/// Generate a unique IPC address for a given test / role combination.
pub fn ipc_addr(test: &str, role: &str) -> String {
    format!(
        "ipc:///tmp/meta_merge_{}_{}_{}",
        test,
        role,
        std::process::id()
    )
}

pub fn make_ingress_conf(name: &str, ipc: &str) -> IngressConfiguration {
    IngressConfiguration {
        name: name.into(),
        socket: SourceConfiguration {
            url: format!("rep+bind:{}", ipc),
            options: Some(SourceOptions {
                receive_timeout: Duration::from_millis(100),
                receive_hwm: 1000,
                topic_prefix_spec: TopicPrefixSpec::None,
                source_cache_size: 100,
                fix_ipc_permissions: None,
                inflight_ops: 100,
            }),
        },
        handler: None,
        eos_policy: Some(EosPolicy::Allow),
    }
}

pub fn make_callbacks(merge_handler: &str) -> CallbacksConfiguration {
    CallbacksConfiguration {
        on_merge: merge_handler.into(),
        on_head_expire: "head_expired_handler".into(),
        on_head_ready: "head_ready_handler".into(),
        on_late_arrival: "late_arrival_handler".into(),
        on_unsupported_message: Some("unsupported_message_handler".into()),
        on_send: Some("send_handler".into()),
    }
}

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

pub fn start_service(
    conf: ServiceConfiguration,
    shutdown: Arc<AtomicBool>,
) -> JoinHandle<Result<()>> {
    let handle = thread::spawn(move || meta_merge::run_service_loop(&conf, Some(shutdown)));
    thread::sleep(Duration::from_millis(500));
    handle
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

    pub fn unsupported() -> Self {
        Self {
            message: Some(Message::user_data(UserData::new("test"))),
            is_eos: false,
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

/// Deterministic pseudo-random delay in `0..window` based on a seed.
/// Uses a single round of a Weyl-sequence hash for fast, repeatable jitter.
pub fn pseudo_delay(seed: u64, window: Duration) -> Duration {
    let hash = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let frac = (hash >> 33) as u32 % (window.as_millis() as u32 + 1);
    Duration::from_millis(frac as u64)
}

/// N frames → EOS → UserData.
pub fn test_steps(num_frames: u32) -> impl Iterator<Item = TestStep> {
    (0..num_frames)
        .map(|_| TestStep::frame())
        .chain(std::iter::once(TestStep::eos()))
        .chain(std::iter::once(TestStep::unsupported()))
}

/// N frames → EOS  (no UserData – used when the batch must be sent before
/// waiting for delivery).
pub fn frame_eos_steps(num_frames: u32) -> impl Iterator<Item = TestStep> {
    (0..num_frames)
        .map(|_| TestStep::frame())
        .chain(std::iter::once(TestStep::eos()))
}

/// Wait (up to `deadline`) for the next message at `reader` accepted by
/// `accept`.  Returns `true` when the predicate matched before the
/// deadline.
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

/// Wait for `unsupported_message_handler` flag to be set, up to `deadline`.
pub fn wait_for_unsupported(deadline: Instant) {
    while Instant::now() < deadline {
        if UNSUPPORTED_HANDLER_CALLED.load(Ordering::SeqCst) {
            return;
        }
        thread::sleep(Duration::from_millis(5));
    }
    panic!("unsupported_message_handler was not called within deadline");
}
