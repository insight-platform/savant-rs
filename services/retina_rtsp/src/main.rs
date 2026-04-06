use anyhow::{anyhow, Result};
use log::{debug, info};
use retina_rtsp::{configuration::ServiceConfiguration, run_service};
use savant_core::transport::zeromq::NonBlockingWriter;
use savant_services_common::job_writer::JobWriter;
use std::{env::args, sync::Arc};
use tokio::sync::Mutex;

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    info!("┌───────────────────────────────────────────────────────┐");
    info!("│                  Retina RTSP Service                  │");
    info!("│ This program is licensed under the APACHE 2.0 license │");
    info!("│      For more information, see the LICENSE file       │");
    info!("│            (c) 2025 BwSoft Management, LLC            │");
    info!("└───────────────────────────────────────────────────────┘");

    let conf_arg = args()
        .nth(1)
        .ok_or_else(|| anyhow!("missing configuration argument"))?;
    info!("Configuration: {}", conf_arg);
    let conf = Arc::new(ServiceConfiguration::new(&conf_arg)?);
    debug!("Configuration: {:?}", conf);

    let groups: Vec<String> = conf.rtsp_sources.keys().cloned().collect();
    let non_blocking_writer = NonBlockingWriter::try_from(&conf.sink)?;
    let sink = Arc::new(Mutex::new(JobWriter::new(non_blocking_writer)));
    let reconnect_interval = conf.reconnect_interval.unwrap();
    let eos_on_restart = conf.eos_on_restart.unwrap_or(true);

    run_service(conf, groups, sink, reconnect_interval, eos_on_restart, None).await
}
