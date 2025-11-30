use actix_web::{get, web, HttpResponse, Responder};
use log::{debug, warn};
use savant_core::json_api::ToSerdeJsonValue;
use savant_core::message::Message;
use serde::Deserialize;
use uuid::Uuid;

use crate::web_service::JobService;

#[derive(Debug, Deserialize)]
struct GetKeyframeQuery {
    source_id: String,
}

fn write_boundary(body: &mut Vec<u8>, boundary: &str, is_final: bool) {
    body.extend_from_slice(b"--");
    body.extend_from_slice(boundary.as_bytes());
    if is_final {
        body.extend_from_slice(b"--");
    }
    body.extend_from_slice(b"\r\n");
}

fn build_multipart_response(
    message: &Message,
    data: &[Vec<u8>],
) -> Result<(String, Vec<u8>), String> {
    let boundary = format!("savant-frame-{}", Uuid::new_v4().as_simple());
    let mut body = Vec::new();

    write_boundary(&mut body, &boundary, false);

    let video_frame = message
        .as_video_frame()
        .ok_or_else(|| "Message is not a VideoFrame".to_string())?;
    let json = video_frame.to_serde_json_value();
    let json_bytes =
        serde_json::to_vec(&json).map_err(|e| format!("JSON serialization failed: {}", e))?;
    body.extend_from_slice(b"Content-Type: application/json\r\n");
    body.extend_from_slice(b"Content-Disposition: inline; name=\"metadata\"\r\n");
    body.extend_from_slice(format!("Content-Length: {}\r\n\r\n", json_bytes.len()).as_bytes());
    body.extend_from_slice(&json_bytes);
    body.extend_from_slice(b"\r\n");

    for (idx, item) in data.iter().enumerate() {
        write_boundary(&mut body, &boundary, false);
        body.extend_from_slice(b"Content-Type: application/octet-stream\r\n");
        body.extend_from_slice(
            format!(
                "Content-Disposition: inline; name=\"data\"; index=\"{}\"\r\n",
                idx
            )
            .as_bytes(),
        );
        body.extend_from_slice(format!("Content-Length: {}\r\n\r\n", item.len()).as_bytes());
        body.extend_from_slice(item);
        body.extend_from_slice(b"\r\n");
    }

    write_boundary(&mut body, &boundary, true);

    Ok((boundary, body))
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
        Ok(Some(frame_data)) => {
            debug!(
                "Returning keyframe: uuid={}, data={}",
                uuid,
                frame_data.data.len()
            );

            match build_multipart_response(&frame_data.message, &frame_data.data) {
                Ok((boundary, body)) => HttpResponse::Ok()
                    .content_type(format!("multipart/mixed; boundary={}", boundary))
                    .body(body),
                Err(e) => {
                    warn!("Failed to build multipart response: {}", e);
                    HttpResponse::InternalServerError().json(serde_json::json!({
                        "error": "Failed to build response",
                        "details": e
                    }))
                }
            }
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
