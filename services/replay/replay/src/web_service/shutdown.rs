use actix_web::{post, web, Responder};
use log::info;

use crate::web_service::{JobService, ResponseMessage};

#[post("/shutdown")]
async fn shutdown(js: web::Data<JobService>) -> impl Responder {
    let mut js_bind = js.shutdown.lock().await;
    info!("Shutting down");
    *js_bind = true;
    ResponseMessage::Ok
}
