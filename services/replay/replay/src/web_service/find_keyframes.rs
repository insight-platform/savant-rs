use actix_web::{post, web, Responder};
use log::debug;
use serde::{Deserialize, Serialize};

use crate::web_service::{JobService, ResponseMessage};

#[derive(Debug, Serialize, Deserialize)]
struct FindKeyframesQuery {
    source_id: String,
    from: Option<u64>,
    to: Option<u64>,
    limit: usize,
}

#[post("/keyframes/find")]
async fn find_keyframes(
    js: web::Data<JobService>,
    query: web::Json<FindKeyframesQuery>,
) -> impl Responder {
    let mut js_bind = js.service.lock().await;
    let uuids_res = js_bind
        .find_keyframes(&query.source_id, query.from, query.to, query.limit)
        .await;
    debug!(
        "Received Keyframe Lookup Query: {}",
        serde_json::to_string(&query).unwrap()
    );
    match uuids_res {
        Ok(uuids) => ResponseMessage::FindKeyframes(
            query.source_id.clone(),
            uuids.into_iter().map(String::from).collect(),
        ),
        Err(e) => ResponseMessage::Error(e.to_string()),
    }
}
