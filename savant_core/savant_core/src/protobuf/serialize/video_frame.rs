use crate::primitives::frame::{
    VideoFrame, VideoFrameContent, VideoFrameProxy, VideoFrameTranscodingMethod,
    VideoFrameTransformation,
};
use crate::primitives::object::VideoObject;
use crate::primitives::video_codec::VideoCodec;
use crate::primitives::Attribute;
use crate::protobuf::serialize::Error;
use hashbrown::{HashMap, HashSet};
use prost::UnknownEnumValue;
use savant_protobuf::generated;
use savant_protobuf::generated::VideoCodec as PbVideoCodec;
use std::str::FromStr;
use std::sync::Arc;
use uuid::Uuid;

fn pb_rational32(num: i64, den: i64) -> generated::Rational32 {
    generated::Rational32 {
        numerator: num.clamp(i32::MIN as i64, i32::MAX as i64) as i32,
        denominator: den.clamp(i32::MIN as i64, i32::MAX as i64) as i32,
    }
}

fn fps_from_pb(fps: &Option<generated::Rational32>) -> (i64, i64) {
    match fps {
        Some(r) if r.numerator > 0 && r.denominator > 0 => {
            (i64::from(r.numerator), i64::from(r.denominator))
        }
        _ => (30, 1),
    }
}

fn time_base_from_pb(tb: &Option<generated::Rational32>) -> (i64, i64) {
    match tb {
        None => (0, 0),
        Some(r) => (i64::from(r.numerator), i64::from(r.denominator)),
    }
}

fn pb_nanoseconds_u128(v: u128) -> generated::NanosecondsU128 {
    generated::NanosecondsU128 {
        high: (v >> 64) as u64,
        low: (v & 0xFFFFFFFFFFFFFFFF) as u64,
    }
}

fn nanoseconds_u128_from_pb(t: &Option<generated::NanosecondsU128>) -> u128 {
    match t {
        None => 0,
        Some(n) => ((n.high as u128) << 64) | (n.low as u128),
    }
}

fn pb_codec(codec: Option<VideoCodec>) -> i32 {
    match codec {
        None => PbVideoCodec::Unspecified as i32,
        Some(VideoCodec::H264) => PbVideoCodec::H264 as i32,
        Some(VideoCodec::Hevc) => PbVideoCodec::Hevc as i32,
        Some(VideoCodec::Jpeg) => PbVideoCodec::Jpeg as i32,
        Some(VideoCodec::SwJpeg) => PbVideoCodec::Swjpeg as i32,
        Some(VideoCodec::Av1) => PbVideoCodec::Av1 as i32,
        Some(VideoCodec::Png) => PbVideoCodec::Png as i32,
        Some(VideoCodec::Vp8) => PbVideoCodec::Vp8 as i32,
        Some(VideoCodec::Vp9) => PbVideoCodec::Vp9 as i32,
        Some(VideoCodec::RawRgba) => PbVideoCodec::RawRgba as i32,
        Some(VideoCodec::RawRgb) => PbVideoCodec::RawRgb as i32,
        Some(VideoCodec::RawNv12) => PbVideoCodec::RawNv12 as i32,
    }
}

fn rust_codec_from_pb(v: i32) -> Result<Option<VideoCodec>, Error> {
    let pb =
        PbVideoCodec::try_from(v).map_err(|e: UnknownEnumValue| Error::EnumConversionError(e.0))?;
    Ok(match pb {
        PbVideoCodec::Unspecified => None,
        PbVideoCodec::H264 => Some(VideoCodec::H264),
        PbVideoCodec::Hevc => Some(VideoCodec::Hevc),
        PbVideoCodec::Jpeg => Some(VideoCodec::Jpeg),
        PbVideoCodec::Swjpeg => Some(VideoCodec::SwJpeg),
        PbVideoCodec::Av1 => Some(VideoCodec::Av1),
        PbVideoCodec::Png => Some(VideoCodec::Png),
        PbVideoCodec::Vp8 => Some(VideoCodec::Vp8),
        PbVideoCodec::Vp9 => Some(VideoCodec::Vp9),
        PbVideoCodec::RawRgba => Some(VideoCodec::RawRgba),
        PbVideoCodec::RawRgb => Some(VideoCodec::RawRgb),
        PbVideoCodec::RawNv12 => Some(VideoCodec::RawNv12),
    })
}

impl From<&VideoFrameProxy> for generated::VideoFrame {
    fn from(vfp: &VideoFrameProxy) -> Self {
        let bind = vfp.get_inner();
        let o = bind.read();
        generated::VideoFrame::from(&*o)
    }
}

impl From<&Box<VideoFrame>> for generated::VideoFrame {
    fn from(video_frame: &Box<VideoFrame>) -> Self {
        let objects = video_frame
            .get_objects()
            .values()
            .map(generated::VideoObject::from)
            .collect();

        generated::VideoFrame {
            previous_frame_seq_id: video_frame.previous_frame_seq_id,
            previous_keyframe: video_frame
                .previous_keyframe
                .map(|ku| Uuid::from_u128(ku).to_string()),
            source_id: video_frame.source_id.clone(),
            uuid: Uuid::from_u128(video_frame.uuid).to_string(),
            creation_timestamp_ns: Some(pb_nanoseconds_u128(video_frame.creation_timestamp_ns)),
            fps: Some(pb_rational32(video_frame.fps.0, video_frame.fps.1)),
            width: video_frame.width,
            height: video_frame.height,
            transcoding_method: generated::VideoFrameTranscodingMethod::from(
                &video_frame.transcoding_method,
            ) as i32,
            video_codec: pb_codec(video_frame.codec),
            keyframe: video_frame.keyframe,
            time_base: Some(pb_rational32(
                video_frame.time_base.0,
                video_frame.time_base.1,
            )),
            pts: video_frame.pts,
            dts: video_frame.dts,
            duration: video_frame.duration,
            attributes: video_frame
                .attributes
                .iter()
                .filter(|a| a.is_persistent)
                .map(|a| a.into())
                .collect(),
            objects,
            content: Some((&*video_frame.content).into()),
            transformations: video_frame
                .transformations
                .iter()
                .map(generated::VideoFrameTransformation::from)
                .collect(),
        }
    }
}

impl TryFrom<&generated::VideoFrame> for VideoFrame {
    type Error = Error;

    fn try_from(value: &generated::VideoFrame) -> Result<Self, Self::Error> {
        let transformations = value
            .transformations
            .iter()
            .filter_map(|t| VideoFrameTransformation::try_from(t).ok())
            .collect::<Vec<VideoFrameTransformation>>();

        let attributes = value
            .attributes
            .iter()
            .map(Attribute::try_from)
            .collect::<Result<Vec<Attribute>, _>>()?;

        let objects = value
            .objects
            .iter()
            .map(|o| VideoObject::try_from(o).map(|vo| (vo.id, vo)))
            .collect::<Result<HashMap<i64, _>, _>>()?;

        let object_parents = value
            .objects
            .iter()
            .flat_map(|o| o.parent_id)
            .collect::<HashSet<i64>>();

        for parent_id in object_parents {
            if !objects.contains_key(&parent_id) {
                return Err(Error::InvalidVideoFrameParentObject(parent_id));
            }
        }

        let max_object_id = objects.keys().max().copied().unwrap_or_default();

        Ok(VideoFrame {
            previous_frame_seq_id: value.previous_frame_seq_id,
            previous_keyframe: value
                .previous_keyframe
                .as_ref()
                .map(|ku| Uuid::from_str(ku).map(|u| u.as_u128()))
                .transpose()?,
            source_id: value.source_id.clone(),
            uuid: Uuid::from_str(&value.uuid)?.as_u128(),
            creation_timestamp_ns: nanoseconds_u128_from_pb(&value.creation_timestamp_ns),
            fps: fps_from_pb(&value.fps),
            width: value.width,
            height: value.height,
            transcoding_method: VideoFrameTranscodingMethod::from(
                &value
                    .transcoding_method
                    .try_into()
                    .map_err(|e: UnknownEnumValue| Error::EnumConversionError(e.0))?,
            ),
            codec: rust_codec_from_pb(value.video_codec)?,
            keyframe: value.keyframe,
            time_base: time_base_from_pb(&value.time_base),
            pts: value.pts,
            dts: value.dts,
            duration: value.duration,
            content: Arc::new(VideoFrameContent::from(value.content.as_ref().unwrap())),
            transformations,
            attributes,
            objects,
            max_object_id,
        })
    }
}

impl TryFrom<&generated::VideoFrame> for VideoFrameProxy {
    type Error = Error;

    fn try_from(value: &generated::VideoFrame) -> Result<Self, Self::Error> {
        Ok(VideoFrameProxy::from_inner(VideoFrame::try_from(value)?))
    }
}

#[cfg(test)]
mod tests {
    use crate::json_api::ToSerdeJsonValue;
    use crate::primitives::frame::VideoFrameProxy;
    use crate::test::gen_frame;
    use savant_protobuf::generated;

    #[test]
    fn test_video_frame() {
        let frame = gen_frame();
        let pattern = 0x10213243_54657687_98A9BACB_DCEDFE0F_u128;
        frame.inner.write().creation_timestamp_ns = pattern;
        let serialized = generated::VideoFrame::from(&frame);
        let restored = VideoFrameProxy::try_from(&serialized).unwrap();
        assert_eq!(restored.inner.read().creation_timestamp_ns, pattern);
        assert_eq!(frame.to_serde_json_value(), restored.to_serde_json_value());
    }
}
