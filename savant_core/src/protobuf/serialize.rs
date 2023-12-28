use std::convert::Infallible;

mod attribute;
mod bounding_box;
mod intersection_kind;
mod message_envelope;
mod polygonal_area;
mod user_data;
mod video_frame;
mod video_frame_batch;
mod video_frame_content;
mod video_frame_transcoding_method;
mod video_frame_transformation;
mod video_frame_update;
mod video_object;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Failed to decode protobuf message: {0}")]
    ProstDecode(prost::DecodeError),
    #[error("Failed to encode protobuf message: {0}")]
    ProstEncode(prost::EncodeError),
    #[error("Failed to parse UUID: {0}")]
    UuidParse(uuid::Error),
    #[error("An object has parent {0} which does not belong to the same frame")]
    InvalidVideoFrameParentObject(i64),
}

impl From<uuid::Error> for Error {
    fn from(error: uuid::Error) -> Self {
        Self::UuidParse(error)
    }
}

impl From<prost::DecodeError> for Error {
    fn from(error: prost::DecodeError) -> Self {
        Self::ProstDecode(error)
    }
}

impl From<prost::EncodeError> for Error {
    fn from(error: prost::EncodeError) -> Self {
        Self::ProstEncode(error)
    }
}

impl From<Infallible> for Error {
    fn from(infallible: Infallible) -> Self {
        match infallible {}
    }
}
