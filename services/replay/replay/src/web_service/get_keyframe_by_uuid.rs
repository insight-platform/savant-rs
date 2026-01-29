use actix_web::{get, web, HttpResponse, Responder};
use log::{debug, warn};
use savant_core::json_api::ToSerdeJsonValue;
use savant_core::message::Message;
use savant_core::protobuf::serialize as protobuf_serialize;
use serde::Deserialize;
use uuid::Uuid;

use crate::web_service::JobService;

#[derive(Debug, Deserialize, Default, Clone, Copy, PartialEq)]
#[serde(rename_all = "lowercase")]
enum MetadataFormat {
    Json,
    #[default]
    Native,
}

#[derive(Debug, Deserialize)]
struct GetKeyframeQuery {
    source_id: String,
    #[serde(default)]
    metadata_format: MetadataFormat,
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
    metadata_format: MetadataFormat,
) -> Result<(String, Vec<u8>), String> {
    let boundary = format!("savant-frame-{}", Uuid::new_v4().as_simple());
    let mut body = Vec::new();

    write_boundary(&mut body, &boundary, false);

    // Serialize metadata based on format
    let (content_type, metadata_bytes) = match metadata_format {
        MetadataFormat::Json => {
            let video_frame = message
                .as_video_frame()
                .ok_or_else(|| "Message is not a VideoFrame".to_string())?;
            let json = video_frame.to_serde_json_value();
            let json_bytes = serde_json::to_vec(&json)
                .map_err(|e| format!("JSON serialization failed: {}", e))?;
            ("application/json", json_bytes)
        }
        MetadataFormat::Native => {
            let native_bytes = protobuf_serialize(message)
                .map_err(|e| format!("Protobuf serialization failed: {}", e))?;
            ("application/x-protobuf", native_bytes)
        }
    };

    body.extend_from_slice(format!("Content-Type: {}\r\n", content_type).as_bytes());
    body.extend_from_slice(b"Content-Disposition: inline; name=\"metadata\"\r\n");
    body.extend_from_slice(format!("Content-Length: {}\r\n\r\n", metadata_bytes.len()).as_bytes());
    body.extend_from_slice(&metadata_bytes);
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
        "Keyframe query: uuid={}, source_id={}, metadata_format={:?}",
        uuid_str, query.source_id, query.metadata_format
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
                "Returning keyframe: uuid={}, data={}, metadata_format={:?}",
                uuid,
                frame_data.data.len(),
                query.metadata_format
            );

            match build_multipart_response(
                &frame_data.message,
                &frame_data.data,
                query.metadata_format,
            ) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use savant_core::protobuf::deserialize as protobuf_deserialize;
    use savant_core::test::gen_frame;

    #[test]
    fn test_build_multipart_response_native_format() {
        let frame = gen_frame();
        let message = frame.to_message();
        let data = vec![vec![1u8, 2, 3, 4], vec![5u8, 6, 7, 8]];

        let result = build_multipart_response(&message, &data, MetadataFormat::Native);
        assert!(result.is_ok());

        let (boundary, body) = result.unwrap();
        let body_str = String::from_utf8_lossy(&body);

        assert!(boundary.starts_with("savant-frame-"));
        assert!(body_str.contains("Content-Type: application/x-protobuf"));
    }

    #[test]
    fn test_native_format_roundtrip() {
        let frame = gen_frame();
        let original_message = frame.to_message();

        let native_bytes = protobuf_serialize(&original_message).unwrap();
        let restored_message = protobuf_deserialize(&native_bytes).unwrap();

        assert!(restored_message.is_video_frame());
        let restored_frame = restored_message.as_video_frame().unwrap();

        assert_eq!(frame.get_uuid(), restored_frame.get_uuid());
        assert_eq!(frame.get_source_id(), restored_frame.get_source_id());
    }
}
