use std::sync::{Arc, OnceLock};
use std::time::SystemTime;
use tokio::sync::Mutex;

use crate::get_or_init_async_runtime;
use crate::metric::user_metric_collector::UserMetricCollector;
use crate::metric::{get_counter, get_gauge, new_counter, new_gauge};
use crate::pipeline::implementation;
use actix_web::{get, post, web, App, HttpResponse, HttpServer, Responder};
use lazy_static::lazy_static;
use log::{debug, error, info};
use prometheus_client::encoding::text::encode;
use prometheus_client::registry::Unit;
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
    stats: Arc<Mutex<Vec<Arc<implementation::Pipeline>>>>,
    status: Arc<Mutex<GstPipelineStatus>>,
    shutdown_token: Arc<OnceLock<String>>,
    shutdown_status: Arc<OnceLock<bool>>,
}

impl WsData {
    pub fn new() -> Self {
        WsData {
            stats: Arc::new(Mutex::new(Vec::new())),
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

pub(crate) fn register_pipeline(pipeline: Arc<implementation::Pipeline>) {
    let runtime = get_or_init_async_runtime();
    let stats = WS_DATA.stats.clone();
    runtime.block_on(async move {
        let mut bind = stats.lock().await;
        bind.push(pipeline);
        info!("Pipeline registered in stats.");
    });
}

pub(crate) fn unregister_pipeline(pipeline: Arc<implementation::Pipeline>) {
    let runtime = get_or_init_async_runtime();
    let stats = WS_DATA.stats.clone();
    runtime.block_on(async move {
        let mut bind = stats.lock().await;
        let prev_len = bind.len();
        debug!("Removing pipeline from stats.");
        bind.retain(|p| !Arc::ptr_eq(p, &pipeline));
        if bind.len() == prev_len {
            error!("Failed to remove pipeline from stats.");
        }
    });
}

pub fn set_status(s: GstPipelineStatus) {
    WS_DATA.set_status(s);
}

pub async fn get_status() -> GstPipelineStatus {
    let s = WS_DATA.status.lock().await;
    s.clone()
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

#[get("/metrics")]
async fn metrics_handler() -> HttpResponse {
    let mut c = get_counter("metric_counter");
    let mut g = get_gauge("metric_gauge");
    let content_type = "application/openmetrics-text; version=1.0.0; charset=utf-8";
    if c.is_none() {
        c = Some(new_counter(
            "metric_counter",
            Some("Counter for metrics"),
            &["label1", "label2"],
            Some(Unit::Other(String::from("Number"))),
        ));
        g = Some(new_gauge(
            "metric_gauge",
            Some("Gauge for metrics"),
            &["label3", "label3"],
            Some(Unit::Other(String::from("Time"))),
        ));
    }
    c.map(|v| v.lock().inc(1, &[&"value1", &"value2"]));
    let unix_time_now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();
    g.map(|v| v.lock().set(unix_time_now, &[&"value3", &"value4"]));
    let mut registry = prometheus_client::registry::Registry::default();
    let boxed_collector = Box::new(UserMetricCollector);
    registry.register_collector(boxed_collector);
    let mut body = String::new();
    if let Err(e) = encode(&mut body, &registry) {
        error!("Failed to encode metrics: {}", e);
        return HttpResponse::InternalServerError()
            .content_type(content_type)
            .finish();
    }
    HttpResponse::Ok().content_type(content_type).body(body)
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
                .service(shutdown_handler)
                .service(metrics_handler);
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
        thread::sleep(Duration::from_millis(200));
        set_status(GstPipelineStatus::Running);

        let client = reqwest::Client::new();
        let r = rt.block_on(
            client
                .post("http://localhost:8888/api/v1/shutdown/12345")
                .send(),
        )?;
        assert_eq!(r.status(), 200);
        stop_webserver();
        Ok(())
    }

    #[test]
    #[serial_test::serial]
    fn test_webserver_metrics() -> anyhow::Result<()> {
        let rt = get_or_init_async_runtime();
        init_webserver(8888)?;
        thread::sleep(Duration::from_millis(200));
        set_status(GstPipelineStatus::Running);

        let client = reqwest::Client::new();
        let r = rt.block_on(client.get("http://localhost:8888/api/v1/metrics").send())?;
        assert_eq!(r.status(), 200);
        let text = rt.block_on(r.text())?;
        assert!(text.contains("metric_counter_Number_total"));
        assert!(text.contains("metric_gauge_Time"));
        stop_webserver();
        Ok(())
    }
}
