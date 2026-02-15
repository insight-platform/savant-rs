use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

use meta_merge::configuration::{
    CallbacksConfiguration, CommonConfiguration, EgressConfiguration, EosPolicy,
    IngressConfiguration, QueueConfiguration, ServiceConfiguration,
};
use savant_core::message::Message;
use savant_core::primitives::attribute_value::{AttributeValue, AttributeValueVariant};
use savant_core::primitives::frame::VideoFrameProxy;
use savant_core::primitives::WithAttributes;
use savant_core::test::gen_frame;
use savant_core::transport::zeromq::{
    ReaderConfig, ReaderResult, SyncReader, SyncWriter, WriterConfig,
};
use savant_core_py::REGISTERED_HANDLERS;
use savant_services_common::job_writer::SinkConfiguration;
use savant_services_common::source::{SourceConfiguration, SourceOptions, TopicPrefixSpec};

const NUM_FRAMES: u32 = 5000;

// ═══════════════════════════════════════════════════════════════════════════
// Timing helpers
// ═══════════════════════════════════════════════════════════════════════════

fn now_nanos() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as i64
}

fn get_int_attr(f: &VideoFrameProxy, namespace: &str, name: &str) -> i64 {
    let attr = f
        .get_attribute(namespace, name)
        .unwrap_or_else(|| panic!("missing ({}, {})", namespace, name));
    match attr.values.as_ref()[0].get() {
        AttributeValueVariant::Integer(n) => *n,
        other => panic!(
            "expected integer for ({}, {}), got {:?}",
            namespace, name, other
        ),
    }
}

fn print_timing_stats(label: &str, nanos: &[i64]) {
    let mut sorted = nanos.to_vec();
    sorted.sort();
    let n = sorted.len();
    let sum: i64 = sorted.iter().sum();
    let mean = sum / n as i64;
    let p50 = sorted[n / 2];
    let p95 = sorted[n * 95 / 100];
    let p99 = sorted[n * 99 / 100];
    let min = sorted[0];
    let max = sorted[n - 1];
    println!(
        "  {}: min={:.1}µs mean={:.1}µs p50={:.1}µs p95={:.1}µs p99={:.1}µs max={:.1}µs",
        label,
        min as f64 / 1000.0,
        mean as f64 / 1000.0,
        p50 as f64 / 1000.0,
        p95 as f64 / 1000.0,
        p99 as f64 / 1000.0,
        max as f64 / 1000.0,
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Minimal PyO3 handlers with timestamp instrumentation
// ═══════════════════════════════════════════════════════════════════════════

#[pyclass]
struct BenchMergeHandler;

#[pymethods]
impl BenchMergeHandler {
    #[new]
    fn new() -> Self {
        Self
    }
    fn __call__(
        &self,
        _ingress_name: &str,
        _topic: &str,
        current_state: Bound<'_, PyAny>,
        _incoming_state: Option<Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        let mut vf: savant_core_py::primitives::frame::VideoFrame =
            current_state.getattr("video_frame")?.extract()?;
        vf.0.set_persistent_attribute(
            "bench",
            "t_merge",
            &None,
            false,
            vec![AttributeValue::integer(now_nanos(), None)],
        );
        Ok(true)
    }
}

#[pyclass]
struct BenchHeadReadyHandler;

#[pymethods]
impl BenchHeadReadyHandler {
    #[new]
    fn new() -> Self {
        Self
    }
    fn __call__(&self, state: Bound<'_, PyAny>) -> PyResult<Option<Py<PyAny>>> {
        let py = state.py();
        let mut vf: savant_core_py::primitives::frame::VideoFrame =
            state.getattr("video_frame")?.extract()?;
        vf.0.set_persistent_attribute(
            "bench",
            "t_ready",
            &None,
            false,
            vec![AttributeValue::integer(now_nanos(), None)],
        );
        let msg = savant_core_py::primitives::message::Message::video_frame(&vf);
        Ok(Some(msg.into_pyobject(py)?.unbind().into()))
    }
}

#[pyclass]
struct BenchHeadExpiredHandler;

#[pymethods]
impl BenchHeadExpiredHandler {
    #[new]
    fn new() -> Self {
        Self
    }
    fn __call__(&self, state: Bound<'_, PyAny>) -> PyResult<Option<Py<PyAny>>> {
        let py = state.py();
        let vf: savant_core_py::primitives::frame::VideoFrame =
            state.getattr("video_frame")?.extract()?;
        let msg = savant_core_py::primitives::message::Message::video_frame(&vf);
        Ok(Some(msg.into_pyobject(py)?.unbind().into()))
    }
}

#[pyclass]
struct BenchLateArrivalHandler;

#[pymethods]
impl BenchLateArrivalHandler {
    #[new]
    fn new() -> Self {
        Self
    }
    fn __call__(&self, _state: Bound<'_, PyAny>) -> PyResult<()> {
        Ok(())
    }
}

#[pyclass]
struct BenchSendHandler;

#[pymethods]
impl BenchSendHandler {
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
        Ok(None)
    }
}

#[pyclass]
struct BenchUnsupportedHandler;

#[pymethods]
impl BenchUnsupportedHandler {
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
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Setup helpers
// ═══════════════════════════════════════════════════════════════════════════

static PYTHON_INIT: std::sync::Once = std::sync::Once::new();

fn init_python() {
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

fn register_handlers() {
    Python::attach(|py| {
        let merge = Py::new(py, BenchMergeHandler::new()).unwrap();
        let head_ready = Py::new(py, BenchHeadReadyHandler::new()).unwrap();
        let head_expired = Py::new(py, BenchHeadExpiredHandler::new()).unwrap();
        let late_arrival = Py::new(py, BenchLateArrivalHandler::new()).unwrap();
        let send = Py::new(py, BenchSendHandler::new()).unwrap();
        let unsupported = Py::new(py, BenchUnsupportedHandler::new()).unwrap();

        let mut h = REGISTERED_HANDLERS.write();
        h.insert("bench_merge".into(), merge.into_any());
        h.insert("bench_head_ready".into(), head_ready.into_any());
        h.insert("bench_head_expired".into(), head_expired.into_any());
        h.insert("bench_late_arrival".into(), late_arrival.into_any());
        h.insert("bench_send".into(), send.into_any());
        h.insert("bench_unsupported".into(), unsupported.into_any());
    });
}

fn ipc_addr(role: &str) -> String {
    format!(
        "ipc:///tmp/meta_merge_bench_{}_{}",
        role,
        std::process::id()
    )
}

fn make_conf(ingress_ipc: &str, egress_ipc: &str) -> ServiceConfiguration {
    ServiceConfiguration {
        ingress: vec![IngressConfiguration {
            name: "bench_ingress".into(),
            socket: SourceConfiguration {
                url: format!("router+bind:{}", ingress_ipc),
                options: Some(SourceOptions {
                    receive_timeout: Duration::from_millis(100),
                    receive_hwm: 10000,
                    topic_prefix_spec: TopicPrefixSpec::None,
                    source_cache_size: 100,
                    fix_ipc_permissions: None,
                    inflight_ops: 100,
                }),
            },
            handler: None,
            eos_policy: Some(EosPolicy::Allow),
        }],
        egress: EgressConfiguration {
            socket: SinkConfiguration {
                url: format!("dealer+connect:{}", egress_ipc),
                options: None,
            },
        },
        common: CommonConfiguration {
            init: None,
            callbacks: CallbacksConfiguration {
                on_merge: "bench_merge".into(),
                on_head_expire: "bench_head_expired".into(),
                on_head_ready: "bench_head_ready".into(),
                on_late_arrival: "bench_late_arrival".into(),
                on_unsupported_message: Some("bench_unsupported".into()),
                on_send: Some("bench_send".into()),
            },
            idle_sleep: Duration::from_millis(1),
            queue: QueueConfiguration {
                max_duration: Duration::from_secs(5),
            },
        },
    }
}

fn make_frame() -> Message {
    let mut frame = gen_frame();
    frame.set_persistent_attribute(
        "bench",
        "t_send",
        &None,
        false,
        vec![AttributeValue::integer(now_nanos(), None)],
    );
    Message::video_frame(&frame)
}

/// Send `count` frames as fast as possible via `SyncWriter`.
fn sender_loop(writer: &SyncWriter, count: u32) {
    for _ in 0..count {
        let mut msg = make_frame();
        writer.send_message("bench", &mut msg, &[]).unwrap();
    }
}

/// Receive exactly `count` video frames via blocking `SyncReader`.
fn receiver_loop(reader: &SyncReader, count: u32) {
    let mut received = 0u32;
    while received < count {
        match reader.receive() {
            Ok(ReaderResult::Message { message, .. }) if message.is_video_frame() => {
                received += 1;
            }
            Ok(_) => {}
            Err(e) => panic!("receiver error: {}", e),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Benchmark
// ═══════════════════════════════════════════════════════════════════════════

fn bench_end_to_end(c: &mut Criterion) {
    init_python();
    register_handlers();

    let shutdown = Arc::new(AtomicBool::new(false));
    let ingress_ipc = ipc_addr("ingress");
    let egress_ipc = ipc_addr("egress");

    let conf = make_conf(&ingress_ipc, &egress_ipc);

    // Dest: blocking SyncReader, router+bind
    let dest_reader = SyncReader::new(
        &ReaderConfig::new()
            .url(&format!("router+bind:{}", egress_ipc))
            .unwrap()
            .with_receive_timeout(5000)
            .unwrap()
            .with_receive_hwm(10000)
            .unwrap()
            .build()
            .unwrap(),
    )
    .unwrap();
    thread::sleep(Duration::from_millis(200));

    // Service
    let service_shutdown = shutdown.clone();
    let service_thread =
        thread::spawn(move || meta_merge::run_service_loop(&conf, Some(service_shutdown)));
    thread::sleep(Duration::from_millis(500));

    // Source: blocking SyncWriter, dealer+connect (fire-and-forget for video frames)
    let source_writer = SyncWriter::new(
        &WriterConfig::new()
            .url(&format!("dealer+connect:{}", ingress_ipc))
            .unwrap()
            .with_send_hwm(10000)
            .unwrap()
            .with_send_timeout(5000)
            .unwrap()
            .build()
            .unwrap(),
    )
    .unwrap();
    thread::sleep(Duration::from_millis(200));

    // ── Warmup: confirm the full pipeline is operational ────────────────
    {
        let mut msg = make_frame();
        source_writer.send_message("bench", &mut msg, &[]).unwrap();
        loop {
            match dest_reader.receive() {
                Ok(ReaderResult::Message { message, .. }) if message.is_video_frame() => break,
                Ok(ReaderResult::Timeout) => continue,
                Ok(_) => continue,
                Err(e) => panic!("warmup receive error: {}", e),
            }
        }
    }

    // ── Criterion throughput benchmark ──────────────────────────────────
    let mut group = c.benchmark_group("end_to_end");
    group.throughput(Throughput::Elements(NUM_FRAMES as u64));
    group.sample_size(10);

    group.bench_function("pipelined_5000", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let start = Instant::now();

                let sw = source_writer.clone();
                let dr = dest_reader.clone();

                let send_handle = thread::spawn(move || sender_loop(&sw, NUM_FRAMES));
                let recv_handle = thread::spawn(move || receiver_loop(&dr, NUM_FRAMES));

                send_handle.join().unwrap();
                recv_handle.join().unwrap();
                total += start.elapsed();
            }
            total
        });
    });

    group.finish();

    // ── Profiling pass: per-stage latency breakdown ────────────────────
    #[derive(Debug)]
    struct FrameTiming {
        ingress_ns: i64,
        process_ns: i64,
        egress_ns: i64,
        total_ns: i64,
    }

    let timings = Arc::new(std::sync::Mutex::new(Vec::with_capacity(
        NUM_FRAMES as usize,
    )));

    let sw = source_writer.clone();
    let send_handle = thread::spawn(move || sender_loop(&sw, NUM_FRAMES));

    let dr = dest_reader.clone();
    let tm = timings.clone();
    let recv_handle = thread::spawn(move || {
        let mut received = 0u32;
        while received < NUM_FRAMES {
            match dr.receive() {
                Ok(ReaderResult::Message { message, .. }) if message.is_video_frame() => {
                    let t_recv = now_nanos();
                    let f = message.as_video_frame().unwrap();
                    let t_send = get_int_attr(&f, "bench", "t_send");
                    let t_merge = get_int_attr(&f, "bench", "t_merge");
                    let t_ready = get_int_attr(&f, "bench", "t_ready");
                    tm.lock().unwrap().push(FrameTiming {
                        ingress_ns: t_merge - t_send,
                        process_ns: t_ready - t_merge,
                        egress_ns: t_recv - t_ready,
                        total_ns: t_recv - t_send,
                    });
                    received += 1;
                }
                Ok(_) => {}
                Err(e) => panic!("profiling receive error: {}", e),
            }
        }
    });

    send_handle.join().unwrap();
    recv_handle.join().unwrap();

    let timings = Arc::try_unwrap(timings).unwrap().into_inner().unwrap();
    let total_wall: Duration = timings
        .iter()
        .map(|t| Duration::from_nanos(t.total_ns as u64))
        .sum();
    let rps = timings.len() as f64 / total_wall.as_secs_f64();

    println!();
    println!("=== Profiling pass ({} frames) ===", NUM_FRAMES);
    print_timing_stats(
        "Ingress (send→merge)",
        &timings.iter().map(|t| t.ingress_ns).collect::<Vec<_>>(),
    );
    print_timing_stats(
        "Process (merge→ready)",
        &timings.iter().map(|t| t.process_ns).collect::<Vec<_>>(),
    );
    print_timing_stats(
        "Egress  (ready→recv)",
        &timings.iter().map(|t| t.egress_ns).collect::<Vec<_>>(),
    );
    print_timing_stats(
        "Total   (send→recv) ",
        &timings.iter().map(|t| t.total_ns).collect::<Vec<_>>(),
    );
    println!("  Effective RPS: {:.0}", rps);

    // ── Cleanup ────────────────────────────────────────────────────────
    shutdown.store(true, Ordering::SeqCst);
    service_thread.join().expect("service thread panicked").ok();
    source_writer.shutdown().ok();
    dest_reader.shutdown().ok();
    REGISTERED_HANDLERS.write().clear();
}

criterion_group!(benches, bench_end_to_end);
criterion_main!(benches);
