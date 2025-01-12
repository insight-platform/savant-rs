use crate::primitives::frame::VideoFrameProxy;
use crate::primitives::frame_batch::VideoFrameBatch;
use crate::primitives::frame_update::VideoFrameUpdate;
use crate::primitives::object::VideoObject;
use crate::primitives::rust::UserData;
use savant_protobuf::generated;
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
    #[error("Failed to convert protobuf enum balue to Rust enum value: {0}")]
    EnumConversionError(i32),
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

pub trait ToProtobuf<'a, T>
where
    T: prost::Message + From<&'a Self>,
    Self: Sized + 'a,
{
    fn to_pb<'b>(&'b self) -> Result<Vec<u8>, Error>
    where
        'b: 'a,
    {
        let pb = T::from(self);
        let mut buf = Vec::new();
        pb.encode(&mut buf)?;
        Ok(buf)
    }
}

pub fn from_pb<T, U>(bytes: &[u8]) -> Result<U, Error>
where
    T: prost::Message + Default,
    U: for<'a> TryFrom<&'a T>,
    Error: for<'a> From<<U as TryFrom<&'a T>>::Error>,
{
    let pb = T::decode(bytes)?;
    let obj = U::try_from(&pb)?;
    Ok(obj)
}

impl ToProtobuf<'_, generated::VideoFrame> for VideoFrameProxy {}
impl ToProtobuf<'_, generated::VideoFrameUpdate> for VideoFrameUpdate {}
impl ToProtobuf<'_, generated::VideoFrameBatch> for VideoFrameBatch {}
impl ToProtobuf<'_, generated::VideoObject> for VideoObject {}
impl ToProtobuf<'_, generated::UserData> for UserData {}

#[cfg(test)]
mod tests {
    use crate::json_api::ToSerdeJsonValue;
    use crate::primitives::rust::VideoFrameProxy;
    use crate::protobuf::serialize::{from_pb, ToProtobuf};
    use crate::test::gen_frame;
    use savant_protobuf::generated;

    #[test]
    fn test() {
        let frame = gen_frame();
        let bytes = frame.to_pb().unwrap();
        let restored_frame = from_pb::<generated::VideoFrame, VideoFrameProxy>(&bytes).unwrap();
        assert_eq!(
            frame.to_serde_json_value(),
            restored_frame.to_serde_json_value()
        );
    }
}
