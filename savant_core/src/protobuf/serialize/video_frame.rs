use crate::primitives::frame::{
    VideoFrame, VideoFrameContent, VideoFrameProxy, VideoFrameTranscodingMethod,
    VideoFrameTransformation,
};
use crate::primitives::object::VideoObject;
use crate::primitives::Attribute;
use crate::protobuf::generated;
use crate::protobuf::serialize::Error;
use hashbrown::{HashMap, HashSet};
use std::str::FromStr;
use std::sync::Arc;
use uuid::Uuid;

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
            source_id: video_frame.source_id.clone(),
            uuid: Uuid::from_u128(video_frame.uuid).to_string(),
            creation_timestamp_ns_high: (video_frame.creation_timestamp_ns >> 64) as u64,
            creation_timestamp_ns_low: (video_frame.creation_timestamp_ns & 0xFFFFFFFFFFFFFFFF)
                as u64,
            framerate: video_frame.framerate.clone(),
            width: video_frame.width,
            height: video_frame.height,
            transcoding_method: generated::VideoFrameTranscodingMethod::from(
                &video_frame.transcoding_method,
            ) as i32,
            codec: video_frame.codec.clone(),
            keyframe: video_frame.keyframe,
            time_base_numerator: video_frame.time_base.0,
            time_base_denominator: video_frame.time_base.1,
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
            .map(VideoFrameTransformation::try_from)
            .collect::<Result<Vec<VideoFrameTransformation>, _>>()?;

        let attributes = value
            .attributes
            .iter()
            .map(Attribute::try_from)
            .collect::<Result<Vec<Attribute>, _>>()?;

        let objects = value
            .objects
            .iter()
            .map(|o| VideoObject::try_from(o).map(|vo| (vo.id, VideoObject::from(vo))))
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
            source_id: value.source_id.clone(),
            uuid: Uuid::from_str(&value.uuid)?.as_u128(),
            creation_timestamp_ns: (value.creation_timestamp_ns_high as u128) << 64
                | (value.creation_timestamp_ns_low as u128),
            framerate: value.framerate.clone(),
            width: value.width,
            height: value.height,
            transcoding_method: VideoFrameTranscodingMethod::from(
                &generated::VideoFrameTranscodingMethod::try_from(value.transcoding_method)?,
            ),
            codec: value.codec.clone(),
            keyframe: value.keyframe,
            time_base: (value.time_base_numerator, value.time_base_denominator),
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
    use crate::protobuf::generated;
    use crate::test::gen_frame;

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
