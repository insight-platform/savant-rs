use actix_web::{get, web, HttpResponse, Responder};
use log::{debug, warn};
use serde::Deserialize;
use uuid::Uuid;

use crate::web_service::JobService;

#[derive(Debug, Deserialize)]
struct GetKeyframeQuery {
    source_id: String,
}

fn codec_to_content_type(codec: Option<&str>) -> &'static str {
    match codec {
        Some("h264") => "video/h264",
        Some("hevc") | Some("h265") => "video/hevc",
        Some("jpeg") => "image/jpeg",
        Some("png") => "image/png",
        Some(c) if c.starts_with("raw-") => "application/octet-stream",
        _ => "application/octet-stream",
    }
}

#[get("/keyframe/{uuid}")]
pub async fn get_keyframe_by_uuid(
    js: web::Data<JobService>,
    uuid_path: web::Path<String>,
    query: web::Query<GetKeyframeQuery>,
) -> impl Responder {
    let uuid_str = uuid_path.into_inner();

    debug!(
        "Keyframe query: uuid={}, source_id={}",
        uuid_str, query.source_id
    );

    let uuid = match Uuid::parse_str(&uuid_str) {
        Ok(u) => u,
        Err(e) => {
            return HttpResponse::BadRequest().json(serde_json::json!({
                "error": "Invalid UUID format",
                "details": e.to_string()
            }));
        }
    };

    if query.source_id.is_empty() {
        return HttpResponse::BadRequest().json(serde_json::json!({
            "error": "Invalid parameter",
            "details": "source_id cannot be empty"
        }));
    }

    let mut js_bind = js.service.lock().await;
    let result = js_bind.get_keyframe_by_uuid(&query.source_id, uuid).await;

    match result {
        Ok(Some(frame)) => {
            let content_type = codec_to_content_type(frame.codec.as_deref());

            debug!(
                "Returning keyframe: uuid={}, timestamp_ns={}, codec={:?}, size={} bytes",
                frame.uuid,
                frame.timestamp_ns,
                frame.codec,
                frame.data.len()
            );

            let mut response = HttpResponse::Ok();
            response
                .content_type(content_type)
                .insert_header(("X-Frame-UUID", frame.uuid.to_string()))
                .insert_header(("X-Frame-Timestamp-NS", frame.timestamp_ns.to_string()))
                .insert_header(("X-Frame-Width", frame.width.to_string()))
                .insert_header(("X-Frame-Height", frame.height.to_string()))
                .insert_header(("X-Frame-Keyframe", frame.keyframe.to_string()))
                .insert_header(("Cache-Control", "public, max-age=3600"));

            if let Some(codec) = frame.codec {
                response.insert_header(("X-Frame-Codec", codec));
            }

            response.body(frame.data)
        }
        Ok(None) => {
            debug!(
                "Keyframe not found: uuid={}, source_id={}",
                uuid, query.source_id
            );

            HttpResponse::NotFound().json(serde_json::json!({
                "error": format!(
                    "Keyframe not found: uuid={} for source_id='{}'",
                    uuid, query.source_id
                )
            }))
        }
        Err(e) => {
            warn!(
                "Error retrieving keyframe: uuid={}, source_id={}, error={}",
                uuid, query.source_id, e
            );

            HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "Internal server error",
                "details": e.to_string()
            }))
        }
    }
}
