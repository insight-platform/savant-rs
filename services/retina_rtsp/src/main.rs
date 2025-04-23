mod configuration;
mod service;
pub(crate) mod utils;

use crate::configuration::ServiceConfiguration;
use anyhow::{anyhow, Result};
use log::{debug, error, info};
use replaydb::job_writer::JobWriter;
use retina::client::SessionGroup;
use savant_core::transport::zeromq::NonBlockingWriter;
use service::run_group;
use std::{env::args, sync::Arc};
use tokio::{sync::Mutex, task::JoinSet};

#[tokio::main]
async fn main() -> Result<()> {
    println!("┌───────────────────────────────────────────────────────┐");
    println!("│                  Retina RTSP Service                  │");
    println!("│ This program is licensed under the APACHE 2.0 license │");
    println!("│      For more information, see the LICENSE file       │");
    println!("│            (c) 2025 BwSoft Management, LLC            │");
    println!("└───────────────────────────────────────────────────────┘");

    env_logger::init();
    let conf_arg = args()
        .nth(1)
        .ok_or_else(|| anyhow!("missing configuration argument"))?;
    info!("Configuration: {}", conf_arg);
    let conf = Arc::new(ServiceConfiguration::new(&conf_arg)?);
    debug!("Configuration: {:?}", conf);

    let rtsp_session_group = Arc::new(SessionGroup::default());

    let non_blocking_writer = NonBlockingWriter::try_from(&conf.sink)?;
    let sink = Arc::new(Mutex::new(JobWriter::new(non_blocking_writer)));

    let mut jobs = JoinSet::new();
    for (group_name, _) in &conf.rtsp_sources {
        let conf = conf.clone();
        // check uniqueness of group_name
        let group_name = group_name.clone();
        let rtsp_session_group = rtsp_session_group.clone();
        let sink = sink.clone();
        jobs.spawn(async move {
            loop {
                tokio::time::sleep(conf.reconnect_interval.unwrap()).await;
                let service_group_res = run_group(
                    conf.clone(),
                    group_name.clone(),
                    rtsp_session_group.clone(),
                    sink.clone(),
                )
                .await;
                if let Err(e) = service_group_res {
                    error!("Error running service group {}: {:?}", group_name, e);
                    continue;
                }
                info!("Service group {} stopped", group_name);
            }
        });
    }

    tokio::signal::ctrl_c().await?;
    rtsp_session_group.await_teardown().await?;
    jobs.abort_all();

    info!("Press Ctrl+C to stop the service");
    Ok(())
}
