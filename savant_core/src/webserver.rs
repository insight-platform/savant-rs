use std::sync::{Arc, OnceLock};
use tokio::sync::Mutex;

use crate::get_or_init_async_runtime;
use crate::rust::FrameProcessingStatRecord;
use actix_web::{get, post, web, App, HttpResponse, HttpServer, Responder};
use lazy_static::lazy_static;
use log::error;
use serde::{Deserialize, Serialize};
use tokio::task::JoinHandle;

#[derive(Deserialize, Serialize, Debug, Clone)]
pub enum GstPipelineStatus {
    #[serde(rename = "initializing")]
    Initializing,
    #[serde(rename = "starting")]
    Starting,
    #[serde(rename = "running")]
    Running,
    #[serde(rename = "stopped")]
    Stopped,
    #[serde(rename = "stopping")]
    Stopping,
    #[serde(rename = "shutdown")]
    Shutdown,
}

struct WsData {
    stats: Arc<Mutex<Option<FrameProcessingStatRecord>>>,
    status: Arc<Mutex<GstPipelineStatus>>,
    shutdown_token: Arc<OnceLock<String>>,
    shutdown_status: Arc<OnceLock<bool>>,
}

impl WsData {
    pub fn new() -> Self {
        WsData {
            stats: Arc::new(Mutex::new(None)),
            status: Arc::new(Mutex::new(GstPipelineStatus::Stopped)),
            shutdown_token: Arc::new(OnceLock::new()),
            shutdown_status: Arc::new(OnceLock::new()),
        }
    }

    pub fn set_status(&self, s: GstPipelineStatus) {
        let runtime = get_or_init_async_runtime();
        let thread_status = self.status.clone();
        let ws_job = WS_JOB.get().expect("Web server job not started");
        if ws_job.is_finished() {
            error!("Web server job is finished unexpectedly, cannot update status.");
        }
        runtime.spawn(async move {
            let mut bind = thread_status.lock().await;
            *bind = s;
        });
    }

    pub fn set_stats(&self, s: FrameProcessingStatRecord) {
        let runtime = get_or_init_async_runtime();
        let thread_stats = self.stats.clone();
        let ws_job = WS_JOB.get().expect("Web server job not started");
        if ws_job.is_finished() {
            error!("Web server job is finished unexpectedly, cannot update stats.");
        }
        runtime.spawn(async move {
            let mut bind = thread_stats.lock().await;
            *bind = Some(s);
        });
    }

    pub async fn get_stats(&self) -> Option<FrameProcessingStatRecord> {
        let bind = self.stats.lock().await;
        bind.clone()
    }

    pub fn set_shutdown_token(&self, token: String) {
        let runtime = get_or_init_async_runtime();
        let thread_token = self.shutdown_token.clone();
        runtime.spawn(async move {
            let val = thread_token.get_or_init(|| token.clone());
            if val != &token {
                error!("Attempted to set shutdown token to a different value.");
            }
        });
    }
}

static WS_JOB: OnceLock<JoinHandle<()>> = OnceLock::new();

lazy_static! {
    static ref WS_DATA: web::Data<WsData> = web::Data::new(WsData::new());
}

pub fn set_status(s: GstPipelineStatus) {
    WS_DATA.set_status(s);
}

pub async fn get_status() -> GstPipelineStatus {
    let s = WS_DATA.status.lock().await;
    s.clone()
}

pub fn set_stats(s: FrameProcessingStatRecord) {
    WS_DATA.set_stats(s);
}

async fn get_stats() -> Option<FrameProcessingStatRecord> {
    WS_DATA.get_stats().await
}

pub fn set_shutdown_token(token: String) {
    WS_DATA.set_shutdown_token(token);
}

fn get_shutdown_token() -> Option<String> {
    WS_DATA.shutdown_token.get().cloned()
}

pub fn get_shutdown_status() -> bool {
    WS_DATA.shutdown_status.get().cloned().unwrap_or(false)
}

pub fn shutdown(status: bool) -> anyhow::Result<()> {
    WS_DATA
        .shutdown_status
        .set(status)
        .map_err(|_| anyhow::anyhow!("Shutdown status already set"))?;
    Ok(())
}

pub fn reset_ws_data() {
    WS_DATA.set_status(GstPipelineStatus::Stopped);
}

#[get("/status")]
async fn status_handler() -> impl Responder {
    let s = get_status().await;
    HttpResponse::Ok().json(s)
}

#[post("/shutdown/{token}")]
async fn shutdown_handler(token: web::Path<String>) -> HttpResponse {
    let shutdown_token = get_shutdown_token();
    if shutdown_token.is_none() {
        return HttpResponse::InternalServerError()
            .body("No shutdown token set. Pipeline shutdown is not supported.");
    } else if shutdown_token.unwrap() != *token {
        return HttpResponse::Unauthorized()
            .body("Invalid shutdown token provided (ignoring the command).");
    } else {
        let res = shutdown(true);
        if res.is_err() {
            return HttpResponse::InternalServerError()
                .body("Failed to set shutdown status multiple times (already set).");
        }
        set_status(GstPipelineStatus::Shutdown);
    }
    HttpResponse::Ok().json("ok")
}

pub fn init_webserver(port: u16) -> anyhow::Result<()> {
    let rt = get_or_init_async_runtime();
    if WS_JOB.get().is_some() {
        return Ok(());
    }
    let job_id = rt.spawn(async move {
        HttpServer::new(move || {
            let scope = web::scope("/api/v1")
                .service(status_handler)
                .service(shutdown_handler);
            App::new().service(scope)
        })
        .bind(("0.0.0.0", port))
        .expect("Failed to bind to host:port")
        .run()
        .await
        .expect("Failed to run server");
        error!("Status web server stopped unexpectedly.");
    });
    WS_JOB.get_or_init(|| job_id);
    Ok(())
}

pub fn stop_webserver() {
    let ws_job = WS_JOB.get().expect("Web server job not started");
    ws_job.abort();
}

#[cfg(test)]
mod tests {
    use crate::get_or_init_async_runtime;
    use crate::webserver::{
        init_webserver, set_shutdown_token, set_status, stop_webserver, GstPipelineStatus,
    };
    use std::thread;
    use std::time::Duration;

    #[test]
    #[serial_test::serial]
    fn test_webserver() -> anyhow::Result<()> {
        // unsafe {
        //     std::env::set_var("RUST_LOG", "debug");
        // }
        // _ = env_logger::try_init();
        init_webserver(8888)?;
        thread::sleep(Duration::from_millis(100));
        set_status(GstPipelineStatus::Running);
        let r = reqwest::blocking::get("http://localhost:8888/api/v1/status")?;
        assert_eq!(r.status(), 200);
        let s: GstPipelineStatus = r.json()?;
        assert!(matches!(s, GstPipelineStatus::Running));
        stop_webserver();
        Ok(())
    }

    #[test]
    #[serial_test::serial]
    fn test_webserver_shutdown() -> anyhow::Result<()> {
        // unsafe {
        //     std::env::set_var("RUST_LOG", "debug");
        // }
        // _ = env_logger::try_init();
        let rt = get_or_init_async_runtime();
        set_shutdown_token("12345".to_string());
        init_webserver(8888)?;
        thread::sleep(Duration::from_millis(100));
        set_status(GstPipelineStatus::Running);

        let client = reqwest::Client::new();
        let r = rt.block_on(
            client
                .post("http://localhost:8888/api/v1/shutdown/12345")
                .send(),
        )?;
        assert_eq!(r.status(), 200);

        let r = reqwest::blocking::get("http://localhost:8888/api/v1/status")?;
        assert_eq!(r.status(), 200);
        let s: GstPipelineStatus = r.json()?;
        assert!(matches!(s, GstPipelineStatus::Shutdown));

        stop_webserver();
        Ok(())
    }
}
