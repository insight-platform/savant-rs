use actix_web::{delete, web, Responder};
use log::{info, warn};
use uuid::Uuid;

use replaydb::service::JobManager;

use crate::web_service::{JobService, ResponseMessage};

#[delete("/job/{id}")]
async fn delete_job(js: web::Data<JobService>, q: web::Path<String>) -> impl Responder {
    let query_uuid_str = q.into_inner();
    let job_uuid = Uuid::try_from(query_uuid_str.as_str());
    if let Err(e) = job_uuid {
        let message = format!("Invalid job UUID: {}, error: {}", query_uuid_str, e);
        return ResponseMessage::Error(message);
    }
    let job_uuid = job_uuid.unwrap();
    info!("Deleting job: {}", &job_uuid);
    let mut js_bind = js.service.lock().await;

    let cleanup = js_bind.clean_stopped_jobs().await;
    if let Err(e) = cleanup {
        return ResponseMessage::Error(e.to_string());
    }

    let res = js_bind.stop_job(job_uuid).await;
    match res {
        Ok(_) => ResponseMessage::Ok,
        Err(e) => {
            warn!("Error stopping job: {}", &e);
            ResponseMessage::Error(e.to_string())
        }
    }
}
