use actix_web::{patch, web, Responder};
use log::info;
use uuid::Uuid;

use replaydb::job::stop_condition::JobStopCondition;
use replaydb::service::JobManager;

use crate::web_service::{JobService, ResponseMessage};

#[patch("/job/{job_id}/stop-condition")]
async fn update_stop_condition(
    js: web::Data<JobService>,
    job_id: web::Path<String>,
    query: web::Json<JobStopCondition>,
) -> impl Responder {
    let query_uuid_str = job_id.into_inner();
    let job_uuid = Uuid::try_from(query_uuid_str.as_str());
    if let Err(e) = job_uuid {
        let message = format!("Invalid job UUID: {}, error: {}", query_uuid_str, e);
        return ResponseMessage::Error(message);
    }
    let job_uuid = job_uuid.unwrap();
    info!(
        "Received the stop condition update request for job {}: {:?}",
        job_uuid, query
    );

    let mut js_bind = js.service.lock().await;
    let job = js_bind.update_stop_condition(job_uuid, query.into_inner());
    match job {
        Ok(_) => ResponseMessage::Ok,
        Err(e) => ResponseMessage::Error(e.to_string()),
    }
}
