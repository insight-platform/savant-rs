use anyhow::{anyhow, Result};
use log::{debug, info};
use retina_rtsp::{configuration::ServiceConfiguration, run_service};
use std::{env::args, sync::Arc};

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

    run_service(conf, None).await
}
