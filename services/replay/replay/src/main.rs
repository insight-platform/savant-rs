use std::env::args;

use actix_web::{web, App, HttpServer};
use anyhow::{anyhow, Result};
use log::{debug, info};
use tokio::sync::Mutex;

use replaydb::service::configuration::ServiceConfiguration;
use replaydb::service::rocksdb_service::RocksDbService;
use replaydb::service::JobManager;

use crate::web_service::del_job::delete_job;
use crate::web_service::find_keyframes::find_keyframes;
use crate::web_service::list_jobs::{list_job, list_jobs, list_stopped_jobs};
use crate::web_service::new_job::new_job;
use crate::web_service::shutdown::shutdown;
use crate::web_service::status::status;
use crate::web_service::update_stop_condition::update_stop_condition;
use crate::web_service::JobService;

mod web_service;

#[actix_web::main]
async fn main() -> Result<()> {
    env_logger::init();

    info!("┌───────────────────────────────────────────────────────┐");
    info!("│                Savant Replay Service                  │");
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
    let rocksdb_service = RocksDbService::new(&conf)?;
    debug!("RocksDbService initialized");
    let job_service = web::Data::new(JobService {
        service: Mutex::new(rocksdb_service),
        shutdown: Mutex::new(false),
    });
    let port = conf.common.management_port;

    let http_job_service = job_service.clone();
    let job = tokio::spawn(
        HttpServer::new(move || {
            let scope = web::scope("/api/v1")
                .app_data(http_job_service.clone())
                .service(status)
                .service(shutdown)
                .service(find_keyframes)
                .service(list_stopped_jobs)
                .service(list_job)
                .service(list_jobs)
                .service(delete_job)
                .service(new_job)
                .service(update_stop_condition);

            App::new().service(scope)
        })
        .bind(("0.0.0.0", port))?
        .run(),
    );

    info!(
        "HTTP server started on port {}, API is available under the '/api/v1/' prefix.",
        port
    );

    let signal_job_service = job_service.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.unwrap();
        info!("Ctrl-C received");
        let mut js_bind = signal_job_service.shutdown.lock().await;
        *js_bind = true;
    });

    loop {
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;

        let mut job_service_bind = job_service.service.lock().await;
        job_service_bind.clean_stopped_jobs().await?;
        drop(job_service_bind);

        if *job_service.shutdown.lock().await {
            job.abort();
            let _ = job.await;
            job_service.service.lock().await.shutdown().await?;
            break;
        }
    }
    Ok(())
}
