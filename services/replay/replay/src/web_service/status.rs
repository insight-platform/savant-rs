use actix_web::{get, web, Responder};
use log::error;

use replaydb::service::JobManager;

use crate::web_service::{JobService, ResponseMessage};

#[get("/status")]
async fn status(js: web::Data<JobService>) -> impl Responder {
    let mut js_bind = js.service.lock().await;
    match js_bind.check_stream_processor_finished().await {
        Ok(finished) => {
            if finished {
                ResponseMessage::StatusFinished
            } else {
                ResponseMessage::StatusRunning
            }
        }
        Err(e) => {
            error!("Stream processor finished with error: {}", e);
            ResponseMessage::Error(e.to_string())
        }
    }
}
