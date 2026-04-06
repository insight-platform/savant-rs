pub mod configuration;
pub mod gst_source;
pub mod ntp_sync;
pub mod service;
pub mod syncer;
pub mod utils;

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Duration;

use anyhow::Result;
use log::{error, info};
use retina::client::SessionGroup;
use savant_services_common::job_writer::JobWriter;
use tokio::sync::Mutex;
use tokio::task::JoinSet;

use crate::configuration::{RtspBackend, ServiceConfiguration};
use crate::service::run_group as run_retina_group;

/// Run the retina RTSP service.
///
/// Connects to the specified RTSP source groups, processes incoming video
/// frames, and publishes them via the shared `sink`.
///
/// # Parameters
/// - `conf` — full service configuration (needed by per-group workers).
/// - `groups` — the subset of group names to run.
/// - `sink` — shared writer for publishing frames.
/// - `reconnect_interval` — delay before reconnecting a failed group.
/// - `eos_on_restart` — whether to send EOS when a stream restarts.
/// - `shutdown` — when `Some`, the loop polls the flag every 100 ms and exits
///   cleanly when it is set to `true`. When `None`, the function runs until a
///   Ctrl+C signal is received.
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
        let group = &conf.rtsp_sources[&group_name];
        let conf = conf.clone();
        let sink = sink.clone();
        let backend = group.backend.clone();

        match backend {
            RtspBackend::Retina => {
                let rtsp_session_group = rtsp_session_group.clone();
                jobs.spawn(async move {
                    loop {
                        tokio::time::sleep(reconnect_interval).await;
                        let result = run_retina_group(
                            conf.clone(),
                            group_name.clone(),
                            rtsp_session_group.clone(),
                            sink.clone(),
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
                            conf.clone(),
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
