pub mod configuration;
pub mod gst_source;
pub mod ntp_sync;
pub mod service;
pub mod syncer;
pub mod utils;

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex as StdMutex,
};
use std::time::Duration;

use anyhow::Result;
use hashbrown::HashMap;
use log::{error, info};
use retina::client::SessionGroup;
use savant_core::transport::zeromq::NonBlockingWriter;
use savant_services_common::job_writer::JobWriter;
use tokio::sync::Mutex;
use tokio::task::JoinSet;

use crate::configuration::{RtspBackend, RtspSourceGroup, ServiceConfiguration};
use crate::service::run_group as run_retina_group;

/// Shared state for the retina RTSP service.
///
/// Holds the sink socket, reconnect parameters, and per-group shutdown
/// coordination.  Designed to be wrapped by PyO3 for Python embedding.
pub struct Service {
    sink: Arc<Mutex<JobWriter>>,
    reconnect_interval: Duration,
    eos_on_restart: bool,
    shutdown_flags: Arc<StdMutex<HashMap<String, Arc<AtomicBool>>>>,
    done_notifiers: Arc<StdMutex<HashMap<String, Vec<tokio::sync::oneshot::Sender<()>>>>>,
    rtsp_session_group: Arc<SessionGroup>,
}

impl Service {
    /// Create a new service from the given configuration.
    ///
    /// Opens the ZeroMQ sink socket described in `conf.sink`.
    pub fn new(conf: &ServiceConfiguration) -> Result<Self> {
        let non_blocking_writer = NonBlockingWriter::try_from(&conf.sink)?;
        let sink = Arc::new(Mutex::new(JobWriter::new(non_blocking_writer)));
        Ok(Self {
            sink,
            reconnect_interval: conf.reconnect_interval.unwrap_or(Duration::from_secs(5)),
            eos_on_restart: conf.eos_on_restart.unwrap_or(true),
            shutdown_flags: Arc::new(StdMutex::new(HashMap::new())),
            done_notifiers: Arc::new(StdMutex::new(HashMap::new())),
            rtsp_session_group: Arc::new(SessionGroup::default()),
        })
    }

    /// Run a single RTSP source group.
    ///
    /// Blocks until the group is stopped via [`stop_group`] / [`request_stop`]
    /// or an unrecoverable error occurs.  Automatically reconnects on
    /// transient failures.
    pub async fn run_group(&self, group: &RtspSourceGroup, name: String) -> Result<()> {
        if matches!(group.backend, RtspBackend::Gstreamer) {
            gstreamer::init()?;
        }

        let shutdown = Arc::new(AtomicBool::new(false));
        self.shutdown_flags
            .lock()
            .unwrap()
            .insert(name.clone(), shutdown.clone());

        let result = self
            .run_group_loop(group, name.clone(), shutdown)
            .await;

        // Cleanup: remove from maps and notify waiters.
        self.shutdown_flags.lock().unwrap().remove(&name);
        if let Some(senders) = self.done_notifiers.lock().unwrap().remove(&name) {
            for tx in senders {
                let _ = tx.send(());
            }
        }

        result
    }

    /// Signal a running group to stop.  Returns immediately.
    pub fn request_stop(&self, name: &str) {
        if let Some(flag) = self.shutdown_flags.lock().unwrap().get(name) {
            flag.store(true, Ordering::SeqCst);
        }
    }

    /// Signal a running group to stop and wait for it to finish.
    pub async fn stop_group(&self, name: &str) {
        let rx = {
            let (tx, rx) = tokio::sync::oneshot::channel();
            let mut notifiers = self.done_notifiers.lock().unwrap();
            notifiers.entry(name.to_string()).or_default().push(tx);
            self.request_stop(name);
            rx
        };
        let _ = rx.await;
    }

    /// Stop all running groups and wait for them to finish.
    pub async fn shutdown(&self) {
        let names: Vec<String> = self
            .shutdown_flags
            .lock()
            .unwrap()
            .keys()
            .cloned()
            .collect();

        let mut receivers = Vec::new();
        {
            let mut notifiers = self.done_notifiers.lock().unwrap();
            for name in &names {
                let (tx, rx) = tokio::sync::oneshot::channel();
                notifiers.entry(name.clone()).or_default().push(tx);
                receivers.push(rx);
            }
        }

        for name in &names {
            self.request_stop(name);
        }

        for rx in receivers {
            let _ = rx.await;
        }
    }

    /// Names of currently running groups.
    pub fn running_groups(&self) -> Vec<String> {
        self.shutdown_flags
            .lock()
            .unwrap()
            .keys()
            .cloned()
            .collect()
    }

    // ── private ──────────────────────────────────────────────────────

    async fn run_group_loop(
        &self,
        group: &RtspSourceGroup,
        name: String,
        shutdown: Arc<AtomicBool>,
    ) -> Result<()> {
        loop {
            if shutdown.load(Ordering::SeqCst) {
                info!("Group '{}' shutdown requested", name);
                break;
            }

            let result = match group.backend {
                RtspBackend::Retina => {
                    run_retina_group(
                        group,
                        name.clone(),
                        self.rtsp_session_group.clone(),
                        self.sink.clone(),
                        self.eos_on_restart,
                        self.reconnect_interval,
                    )
                    .await
                }
                RtspBackend::Gstreamer => {
                    gst_source::run_group(
                        group,
                        name.clone(),
                        self.sink.clone(),
                        self.eos_on_restart,
                    )
                    .await
                }
            };

            if let Err(e) = result {
                error!("Error running group '{}': {:?}", name, e);
            } else {
                info!("Group '{}' stopped", name);
            }

            if shutdown.load(Ordering::SeqCst) {
                break;
            }

            // Use tokio::select! so that shutdown interrupts the sleep.
            tokio::select! {
                _ = tokio::time::sleep(self.reconnect_interval) => {}
                _ = async {
                    loop {
                        if shutdown.load(Ordering::SeqCst) {
                            return;
                        }
                        tokio::time::sleep(Duration::from_millis(100)).await;
                    }
                } => {
                    info!("Group '{}' shutdown requested during reconnect wait", name);
                    break;
                }
            }
        }
        Ok(())
    }
}

/// Run the retina RTSP service (legacy free-function API).
///
/// Connects to the specified RTSP source groups, processes incoming video
/// frames, and publishes them via the shared `sink`.
pub async fn run_service(
    conf: Arc<ServiceConfiguration>,
    groups: Vec<String>,
    sink: Arc<Mutex<JobWriter>>,
    reconnect_interval: Duration,
    eos_on_restart: bool,
    shutdown: Option<Arc<AtomicBool>>,
) -> Result<()> {
    let has_gst = groups.iter().any(|name| {
        conf.rtsp_sources
            .get(name)
            .map(|g| matches!(g.backend, RtspBackend::Gstreamer))
            .unwrap_or(false)
    });
    if has_gst {
        gstreamer::init()?;
    }

    let rtsp_session_group = Arc::new(SessionGroup::default());

    let mut jobs = JoinSet::new();
    for group_name in groups {
        let group = conf.rtsp_sources[&group_name].clone();
        let sink = sink.clone();
        let rtsp_session_group = rtsp_session_group.clone();

        match group.backend {
            RtspBackend::Retina => {
                jobs.spawn(async move {
                    loop {
                        tokio::time::sleep(reconnect_interval).await;
                        let result = run_retina_group(
                            &group,
                            group_name.clone(),
                            rtsp_session_group.clone(),
                            sink.clone(),
                            eos_on_restart,
                            reconnect_interval,
                        )
                        .await;
                        if let Err(e) = result {
                            error!("Error running retina group {}: {:?}", group_name, e);
                            continue;
                        }
                        info!("Retina group {} stopped", group_name);
                    }
                });
            }
            RtspBackend::Gstreamer => {
                jobs.spawn(async move {
                    loop {
                        tokio::time::sleep(reconnect_interval).await;
                        let result = gst_source::run_group(
                            &group,
                            group_name.clone(),
                            sink.clone(),
                            eos_on_restart,
                        )
                        .await;
                        if let Err(e) = result {
                            error!("Error running gstreamer group {}: {:?}", group_name, e);
                            continue;
                        }
                        info!("GStreamer group {} stopped", group_name);
                    }
                });
            }
        }
    }

    if let Some(shutdown_flag) = shutdown {
        loop {
            if shutdown_flag.load(Ordering::SeqCst) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    } else {
        tokio::signal::ctrl_c().await?;
    }

    rtsp_session_group.await_teardown().await?;
    jobs.abort_all();

    Ok(())
}
