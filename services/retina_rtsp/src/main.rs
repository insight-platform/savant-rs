use anyhow::{anyhow, Result};
use log::{debug, info};
use retina_rtsp::{configuration::ServiceConfiguration, Service};
use std::env::args;
use tokio::task::JoinSet;

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
    let conf = ServiceConfiguration::new(&conf_arg)?;
    debug!("Configuration: {:?}", conf);

    let service = Service::new(&conf)?;
    let service = std::sync::Arc::new(service);

    let mut jobs = JoinSet::new();
    for (group_name, group) in &conf.rtsp_sources {
        let svc = service.clone();
        let name = group_name.clone();
        let group = group.clone();
        jobs.spawn(async move { svc.run_group(&group, name).await });
    }

    tokio::signal::ctrl_c().await?;
    service.shutdown().await;
    jobs.abort_all();

    Ok(())
}
