use actix_web::body::BoxBody;
use actix_web::http::header::ContentType;
use actix_web::{HttpResponse, Responder};
use serde::Serialize;
use tokio::sync::Mutex;

use replaydb::job::configuration::JobConfiguration;
use replaydb::job::stop_condition::JobStopCondition;
use replaydb::service::rocksdb_service::RocksDbService;

pub mod del_job;
pub mod find_keyframes;
pub mod get_keyframe_by_uuid;
pub mod list_jobs;
pub mod new_job;
pub mod shutdown;
pub mod status;
pub mod update_stop_condition;

pub struct JobService {
    pub service: Mutex<RocksDbService>,
    pub shutdown: Mutex<bool>,
}

#[derive(Debug, Serialize)]
enum ResponseMessage {
    #[serde(rename = "ok")]
    Ok,
    #[serde(rename = "jobs")]
    ListJobs(Vec<(String, JobConfiguration, JobStopCondition)>),
    #[serde(rename = "stopped_jobs")]
    ListStoppedJobs(Vec<(String, JobConfiguration, Option<String>)>),
    #[serde(rename = "new_job")]
    NewJob(String),
    #[serde(rename = "keyframes")]
    FindKeyframes(String, Vec<String>),
    #[serde(rename = "running")]
    StatusRunning,
    #[serde(rename = "finished")]
    StatusFinished,
    #[serde(rename = "error")]
    Error(String),
}

impl Responder for ResponseMessage {
    type Body = BoxBody;
    fn respond_to(self, _req: &actix_web::HttpRequest) -> HttpResponse<Self::Body> {
        let body = serde_json::to_string(&self).unwrap();
        let mut resp = match self {
            ResponseMessage::Error(_) => HttpResponse::InternalServerError(),
            _ => HttpResponse::Ok(),
        };
        resp.content_type(ContentType::json()).body(body)
    }
}
