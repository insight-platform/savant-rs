use crate::primitives::frame::{
    VideoFrame, VideoFrameContent, VideoFrameProxy, VideoFrameTranscodingMethod,
    VideoFrameTransformation,
};
use crate::primitives::object::{VideoObject, VideoObjectProxy};
use crate::primitives::Attribute;
use crate::protobuf::generated;
use crate::protobuf::serialize::Error;
use hashbrown::{HashMap, HashSet};
use parking_lot::RwLock;
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
    fn from(vf: &Box<VideoFrame>) -> Self {
        let objects = vf
            .get_resident_objects()
            .values()
            .map(|o| generated::VideoObject::from(&VideoObjectProxy::from(o.clone())))
            .collect();
        generated::VideoFrame {
            previous_frame_seq_id: vf.previous_frame_seq_id,
            source_id: vf.source_id.clone(),
            uuid: Uuid::from_u128(vf.uuid).to_string(),
            creation_timestamp_ns_high: (vf.creation_timestamp_ns >> 64) as u64,
            creation_timestamp_ns_low: (vf.creation_timestamp_ns & 0xFFFFFFFFFFFFFFFF) as u64,
            framerate: vf.framerate.clone(),
            width: vf.width,
            height: vf.height,
            transcoding_method: generated::VideoFrameTranscodingMethod::from(&vf.transcoding_method)
                as i32,
            codec: vf.codec.clone(),
            keyframe: vf.keyframe,
            time_base_numerator: vf.time_base.0,
            time_base_denominator: vf.time_base.1,
            pts: vf.pts,
            dts: vf.dts,
            duration: vf.duration,
            attributes: vf.attributes.values().map(|a| a.into()).collect(),
            objects,
            content: Some((&vf.content).into()),
            transformations: vf.transformations.iter().map(|t| t.into()).collect(),
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
            .map(|a| {
                let key = (a.namespace.clone(), a.name.clone());
                Attribute::try_from(a).map(|v| (key, v))
            })
            .collect::<Result<HashMap<(String, String), Attribute>, _>>()?;

        let resident_objects = value
            .objects
            .iter()
            .map(|o| VideoObject::try_from(o).map(|vo| (vo.id, Arc::new(RwLock::new(vo)))))
            .collect::<Result<HashMap<i64, Arc<RwLock<VideoObject>>>, _>>()?;

        let object_parents = value
            .objects
            .iter()
            .flat_map(|o| o.parent_id)
            .collect::<HashSet<i64>>();

        for parent_id in object_parents {
            if !resident_objects.contains_key(&parent_id) {
                return Err(Error::InvalidVideoFrameParentObject(parent_id));
            }
        }

        let max_object_id = resident_objects.keys().max().copied().unwrap_or_default();

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
            content: VideoFrameContent::from(value.content.as_ref().unwrap()),
            transformations,
            attributes,
            offline_objects: vec![],
            resident_objects,
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
        let serialized = generated::VideoFrame::from(&frame);
        let restored = VideoFrameProxy::try_from(&serialized).unwrap();
        assert_eq!(frame.to_serde_json_value(), restored.to_serde_json_value());
    }
}
