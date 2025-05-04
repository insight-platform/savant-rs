mod configuration;

use anyhow::{anyhow, Result};
use configuration::ServiceConfiguration;
use log::{debug, info};
use std::{env::args, sync::Arc};

fn main() -> Result<()> {
    println!("┌───────────────────────────────────────────────────────┐");
    println!("│                  Router Service                       │");
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

    Ok(())
}
