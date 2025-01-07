use std::sync::{Arc, OnceLock};
use tokio::sync::Mutex;

use crate::get_or_init_async_runtime;
use crate::metric::user_metric_collector::UserMetricCollector;
use crate::pipeline::implementation;
use actix_web::{get, post, web, App, HttpResponse, HttpServer, Responder};
use lazy_static::lazy_static;
use log::{debug, error, info};
use prometheus_client::encoding::text::encode;
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
    static ref PID: Mutex<i32> = Mutex::new(0);
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

#[cfg(test)]
pub fn shutdown(_status: bool) -> anyhow::Result<()> {
    Ok(())
}

#[cfg(not(test))]
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

#[derive(Deserialize)]
enum ShutdownMode {
    #[serde(rename = "graceful")]
    Notify,
    #[serde(rename = "signal")]
    Signal,
}

#[derive(Deserialize)]
struct ShutdownParams {
    token: String,
    mode: ShutdownMode,
}

#[post("/shutdown/{token}/{mode}")]
async fn shutdown_handler(params: web::Path<ShutdownParams>) -> HttpResponse {
    let shutdown_params: ShutdownParams = params.into_inner();
    let shutdown_token = get_shutdown_token();
    if shutdown_token.is_none() {
        return HttpResponse::InternalServerError()
            .body("No shutdown token set. Pipeline shutdown is not supported.");
    } else if shutdown_token.unwrap() != shutdown_params.token {
        return HttpResponse::Unauthorized()
            .body("Invalid shutdown token provided (ignoring the command).");
    } else {
        let res = shutdown(true);
        if res.is_err() {
            return HttpResponse::InternalServerError()
                .body("Failed to set shutdown status multiple times (already set).");
        }
        set_status(GstPipelineStatus::Shutdown);
        if matches!(shutdown_params.mode, ShutdownMode::Signal) {
            let pid = PID.lock().await;
            _ = nix::sys::signal::kill(
                nix::unistd::Pid::from_raw(*pid),
                nix::sys::signal::Signal::SIGINT,
            );
        }
    }
    HttpResponse::Ok().json("ok")
}

#[get("/metrics")]
async fn metrics_handler() -> HttpResponse {
    let content_type = "application/openmetrics-text; version=1.0.0; charset=utf-8";
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
    let pid = std::process::id() as i32;
    let rt = get_or_init_async_runtime();
    rt.block_on(async {
        let mut bind = PID.lock().await;
        *bind = pid;
    });

    if WS_JOB.get().is_some() {
        return Ok(());
    }
    let job_id = rt.spawn(async move {
        HttpServer::new(move || {
            App::new()
                .service(status_handler)
                .service(shutdown_handler)
                .service(metrics_handler)
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
    use crate::metric::{del_metric, get_or_create_counter, get_or_create_gauge, set_extra_labels};
    use crate::webserver::{
        init_webserver, set_shutdown_token, set_status, stop_webserver, GstPipelineStatus,
    };
    use hashbrown::HashMap;
    use prometheus_client::registry::Unit;
    use std::thread;
    use std::time::{Duration, SystemTime};

    const TOKEN: &str = "12345";

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
        let r = reqwest::blocking::get("http://localhost:8888/status")?;
        assert_eq!(r.status(), 200);
        let s: GstPipelineStatus = r.json()?;
        assert!(matches!(s, GstPipelineStatus::Running));
        stop_webserver();
        Ok(())
    }

    #[test]
    #[serial_test::serial]
    fn test_webserver_shutdown_graceful() -> anyhow::Result<()> {
        // unsafe {
        //     std::env::set_var("RUST_LOG", "debug");
        // }
        // _ = env_logger::try_init();
        let rt = get_or_init_async_runtime();
        set_shutdown_token(TOKEN.to_string());
        init_webserver(8888)?;
        thread::sleep(Duration::from_millis(500));
        set_status(GstPipelineStatus::Running);
        let client = reqwest::Client::new();
        let r = rt.block_on(
            client
                .post("http://localhost:8888/shutdown/12345/graceful")
                .send(),
        )?;
        assert_eq!(r.status(), 200);
        stop_webserver();
        Ok(())
    }

    #[test]
    #[serial_test::serial]
    fn test_webserver_shutdown_signal() -> anyhow::Result<()> {
        let rt = get_or_init_async_runtime();
        set_shutdown_token(TOKEN.to_string());
        init_webserver(8888)?;
        thread::sleep(Duration::from_millis(500));
        set_status(GstPipelineStatus::Running);
        let (snd, rec) = crossbeam::channel::bounded(1);
        ctrlc::set_handler(move || {
            snd.send(()).unwrap();
        })
        .expect("Error setting Ctrl-C handler");
        let client = reqwest::Client::new();
        let r = rt.block_on(
            client
                .post("http://localhost:8888/shutdown/12345/signal")
                .send(),
        )?;
        rec.recv().unwrap();
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
        set_extra_labels(HashMap::from([(
            String::from("hello"),
            String::from("there"),
        )]));

        let c = get_or_create_counter(
            "metric_counter",
            Some("Counter for metrics"),
            &["label1", "label2"],
            Some(Unit::Other(String::from("Number"))),
        );

        let g = get_or_create_gauge(
            "metric_gauge",
            Some("Gauge for metrics"),
            &["label3", "label3"],
            Some(Unit::Other(String::from("Time"))),
        );

        c.lock().inc(1, &[&"value1", &"value2"])?;
        let unix_time_now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        g.lock().set(unix_time_now, &[&"value3", &"value4"])?;

        let client = reqwest::Client::new();
        let r = rt.block_on(client.get("http://localhost:8888/metrics").send())?;
        assert_eq!(r.status(), 200);
        let text = rt.block_on(r.text())?;
        assert!(text.contains("metric_counter_Number_total"));
        assert!(text.contains("metric_gauge_Time"));
        assert!(text.contains("hello"));
        del_metric("metric_counter");
        del_metric("metric_gauge");
        stop_webserver();
        Ok(())
    }
}

// TODO: pipeline metric collector
// TODO: common labels
