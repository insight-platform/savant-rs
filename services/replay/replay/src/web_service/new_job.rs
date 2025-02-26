use actix_web::{put, web, Responder};
use log::info;

use replaydb::job::query::JobQuery;
use replaydb::service::JobManager;

use crate::web_service::{JobService, ResponseMessage};

#[put("/job")]
async fn new_job(js: web::Data<JobService>, query: web::Json<JobQuery>) -> impl Responder {
    info!("Received New Job Query: {:?}", query);
    let mut js_bind = js.service.lock().await;
    let job = js_bind.add_job(query.into_inner()).await;
    match job {
        Ok(job) => ResponseMessage::NewJob(job.to_string()),
        Err(e) => ResponseMessage::Error(e.to_string()),
    }
}
