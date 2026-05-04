use crate::primitives::frame::{
    VideoFrameContent, VideoFrameInner, VideoFrame, VideoFrameTranscodingMethod,
    VideoFrameTransformation,
};
use crate::primitives::misc_track::{MiscTrackCategory, MiscTrackData, MiscTrackFrame, TrackState};
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
        Some(r) if r.numerator > 0 && r.denominator > 0 => {
            (i64::from(r.numerator), i64::from(r.denominator))
        }
        _ => (1, 1_000_000),
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

// ---- misc-track protobuf helpers ----

fn track_state_to_pb(s: TrackState) -> i32 {
    match s {
        TrackState::Empty => generated::TrackState::Empty as i32,
        TrackState::Active => generated::TrackState::Active as i32,
        TrackState::Inactive => generated::TrackState::Inactive as i32,
        TrackState::Tentative => generated::TrackState::Tentative as i32,
        TrackState::Projected => generated::TrackState::Projected as i32,
    }
}

/// Converts an i32 protobuf `TrackState` wire value into `TrackState`.
/// Returns `Error::UnknownTrackState` when the wire enum value is invalid.
/// This hard-fails at the message boundary to prevent silent corruption.
fn track_state_from_pb(v: i32) -> Result<TrackState, Error> {
    let pb = generated::TrackState::try_from(v).map_err(|e| Error::UnknownTrackState(e.0))?;
    Ok(match pb {
        generated::TrackState::Empty => TrackState::Empty,
        generated::TrackState::Active => TrackState::Active,
        generated::TrackState::Inactive => TrackState::Inactive,
        generated::TrackState::Tentative => TrackState::Tentative,
        generated::TrackState::Projected => TrackState::Projected,
    })
}

fn misc_track_category_to_pb(c: MiscTrackCategory) -> i32 {
    match c {
        MiscTrackCategory::Shadow => generated::MiscTrackCategory::MiscTrackShadow as i32,
        MiscTrackCategory::Terminated => generated::MiscTrackCategory::MiscTrackTerminated as i32,
        MiscTrackCategory::PastFrame => generated::MiscTrackCategory::MiscTrackPastFrame as i32,
    }
}

/// Converts an i32 protobuf `MiscTrackCategory` wire value into `MiscTrackCategory`.
/// Returns `Error::UnknownMiscTrackCategory` when the wire enum value is invalid.
/// Rejecting unknown values here avoids silent defaulting/data corruption.
fn misc_track_category_from_pb(v: i32) -> Result<MiscTrackCategory, Error> {
    let pb = generated::MiscTrackCategory::try_from(v)
        .map_err(|e| Error::UnknownMiscTrackCategory(e.0))?;
    Ok(match pb {
        generated::MiscTrackCategory::MiscTrackShadow => MiscTrackCategory::Shadow,
        generated::MiscTrackCategory::MiscTrackTerminated => MiscTrackCategory::Terminated,
        generated::MiscTrackCategory::MiscTrackPastFrame => MiscTrackCategory::PastFrame,
    })
}

fn misc_track_frame_to_pb(f: &MiscTrackFrame) -> generated::MiscTrackFrame {
    generated::MiscTrackFrame {
        frame_num: f.frame_num,
        bbox_left: f.bbox_left,
        bbox_top: f.bbox_top,
        bbox_width: f.bbox_width,
        bbox_height: f.bbox_height,
        confidence: f.confidence,
        age: f.age,
        state: track_state_to_pb(f.state),
        visibility: f.visibility,
    }
}

/// Converts protobuf `MiscTrackFrame` into `MiscTrackFrame`.
/// Validates i32 `state` wire enum and returns `Error::UnknownTrackState` on failure.
/// Failing fast at deserialization avoids silently mapping bad states.
fn misc_track_frame_from_pb(f: &generated::MiscTrackFrame) -> Result<MiscTrackFrame, Error> {
    Ok(MiscTrackFrame {
        frame_num: f.frame_num,
        bbox_left: f.bbox_left,
        bbox_top: f.bbox_top,
        bbox_width: f.bbox_width,
        bbox_height: f.bbox_height,
        confidence: f.confidence,
        age: f.age,
        state: track_state_from_pb(f.state)?,
        visibility: f.visibility,
    })
}

fn misc_track_data_to_pb(d: &MiscTrackData) -> generated::MiscTrackData {
    generated::MiscTrackData {
        object_id: d.object_id,
        class_id: d.class_id,
        label: d.label.clone(),
        source_id: d.source_id.clone(),
        category: misc_track_category_to_pb(d.category),
        frames: d.frames.iter().map(misc_track_frame_to_pb).collect(),
    }
}

/// Converts protobuf `MiscTrackData` into `MiscTrackData`.
/// All numeric fields are `i64` on both wire and domain sides, so only the
/// two enum fields (`category`, frame-level `state`) can fail — returning a
/// typed `Error::UnknownMiscTrackCategory` / `Error::UnknownTrackState`.
fn misc_track_data_from_pb(d: &generated::MiscTrackData) -> Result<MiscTrackData, Error> {
    Ok(MiscTrackData {
        object_id: d.object_id,
        class_id: d.class_id,
        label: d.label.clone(),
        source_id: d.source_id.clone(),
        category: misc_track_category_from_pb(d.category)?,
        frames: d
            .frames
            .iter()
            .map(misc_track_frame_from_pb)
            .collect::<Result<Vec<_>, _>>()?,
    })
}

impl From<&VideoFrame> for generated::VideoFrame {
    fn from(vfp: &VideoFrame) -> Self {
        let bind = vfp.get_inner();
        let o = bind.read();
        generated::VideoFrame::from(&*o)
    }
}

impl From<&Box<VideoFrameInner>> for generated::VideoFrame {
    fn from(video_frame: &Box<VideoFrameInner>) -> Self {
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
            misc_tracks: video_frame
                .misc_tracks
                .iter()
                .map(misc_track_data_to_pb)
                .collect(),
        }
    }
}

impl TryFrom<&generated::VideoFrame> for VideoFrameInner {
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

        Ok(VideoFrameInner {
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
            misc_tracks: value
                .misc_tracks
                .iter()
                .map(misc_track_data_from_pb)
                .collect::<Result<Vec<_>, _>>()?,
            otel_stack: Vec::new(),
        })
    }
}

impl TryFrom<&generated::VideoFrame> for VideoFrame {
    type Error = Error;

    fn try_from(value: &generated::VideoFrame) -> Result<Self, Self::Error> {
        Ok(VideoFrame::from_inner(VideoFrameInner::try_from(
            value,
        )?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json_api::ToSerdeJsonValue;
    use crate::primitives::frame::VideoFrame;
    use crate::test::gen_frame;
    use savant_protobuf::generated;

    #[test]
    fn test_video_frame() {
        let frame = gen_frame();
        let pattern = 0x10213243_54657687_98A9BACB_DCEDFE0F_u128;
        frame.0.write().creation_timestamp_ns = pattern;
        let serialized = generated::VideoFrame::from(&frame);
        let restored = VideoFrame::try_from(&serialized).unwrap();
        assert_eq!(restored.0.read().creation_timestamp_ns, pattern);
        assert_eq!(frame.to_serde_json_value(), restored.to_serde_json_value());
    }

    #[test]
    fn time_base_from_pb_none_returns_default() {
        assert_eq!(time_base_from_pb(&None), (1, 1_000_000));
    }

    #[test]
    fn time_base_from_pb_zero_values_return_default() {
        let zero_both = generated::Rational32 {
            numerator: 0,
            denominator: 0,
        };
        assert_eq!(time_base_from_pb(&Some(zero_both)), (1, 1_000_000));

        let zero_num = generated::Rational32 {
            numerator: 0,
            denominator: 90_000,
        };
        assert_eq!(time_base_from_pb(&Some(zero_num)), (1, 1_000_000));

        let zero_den = generated::Rational32 {
            numerator: 1,
            denominator: 0,
        };
        assert_eq!(time_base_from_pb(&Some(zero_den)), (1, 1_000_000));
    }

    #[test]
    fn time_base_from_pb_negative_values_return_default() {
        let neg_num = generated::Rational32 {
            numerator: -1,
            denominator: 90_000,
        };
        assert_eq!(time_base_from_pb(&Some(neg_num)), (1, 1_000_000));

        let neg_den = generated::Rational32 {
            numerator: 1,
            denominator: -1,
        };
        assert_eq!(time_base_from_pb(&Some(neg_den)), (1, 1_000_000));
    }

    #[test]
    fn time_base_from_pb_valid_values_pass_through() {
        let valid = generated::Rational32 {
            numerator: 1,
            denominator: 90_000,
        };
        assert_eq!(time_base_from_pb(&Some(valid)), (1, 90_000));
    }

    #[test]
    fn test_misc_tracks_protobuf_round_trip() {
        let frame = gen_frame();
        frame.add_misc_tracks(vec![
            MiscTrackData {
                object_id: 42,
                class_id: 1,
                label: Some("person".to_string()),
                source_id: "cam-1".to_string(),
                category: MiscTrackCategory::Shadow,
                frames: vec![MiscTrackFrame {
                    frame_num: 10,
                    bbox_left: 1.0,
                    bbox_top: 2.0,
                    bbox_width: 30.0,
                    bbox_height: 40.0,
                    confidence: 0.9,
                    age: 5,
                    state: TrackState::Tentative,
                    visibility: 0.7,
                }],
            },
            MiscTrackData {
                object_id: 99,
                class_id: 0,
                label: None,
                source_id: "cam-2".to_string(),
                category: MiscTrackCategory::Terminated,
                frames: vec![],
            },
        ]);

        let serialized = generated::VideoFrame::from(&frame);
        assert_eq!(serialized.misc_tracks.len(), 2);

        let restored = VideoFrame::try_from(&serialized).unwrap();
        let tracks = restored.get_misc_tracks();
        assert_eq!(tracks.len(), 2);

        assert_eq!(tracks[0].object_id, 42);
        assert_eq!(tracks[0].class_id, 1);
        assert_eq!(tracks[0].label.as_deref(), Some("person"));
        assert_eq!(tracks[0].category, MiscTrackCategory::Shadow);
        assert_eq!(tracks[0].frames.len(), 1);
        assert_eq!(tracks[0].frames[0].state, TrackState::Tentative);

        assert_eq!(tracks[1].object_id, 99);
        assert_eq!(tracks[1].category, MiscTrackCategory::Terminated);
        assert!(tracks[1].label.is_none());
    }

    #[test]
    fn test_misc_track_class_id_boundary_round_trip() {
        // With i64 everywhere, both extrema round-trip losslessly.
        let frame = gen_frame();
        frame.add_misc_tracks(vec![
            MiscTrackData {
                object_id: 1,
                class_id: i64::MAX,
                label: Some("max".to_string()),
                source_id: "s".to_string(),
                category: MiscTrackCategory::Shadow,
                frames: vec![],
            },
            MiscTrackData {
                object_id: 2,
                class_id: i64::MIN,
                label: Some("min".to_string()),
                source_id: "s".to_string(),
                category: MiscTrackCategory::Shadow,
                frames: vec![],
            },
        ]);
        let pb = generated::VideoFrame::from(&frame);
        let restored = VideoFrame::try_from(&pb).unwrap();
        let tracks = restored.get_misc_tracks();
        assert_eq!(tracks[0].class_id, i64::MAX);
        assert_eq!(tracks[1].class_id, i64::MIN);
    }

    #[test]
    fn test_misc_track_unknown_track_state() {
        let mut pb = generated::VideoFrame::from(&gen_frame());
        pb.misc_tracks = vec![generated::MiscTrackData {
            class_id: 1,
            object_id: 1,
            label: None,
            source_id: "s".into(),
            category: 0,
            frames: vec![generated::MiscTrackFrame {
                state: 42,
                frame_num: 0,
                bbox_left: 0.0,
                bbox_top: 0.0,
                bbox_width: 0.0,
                bbox_height: 0.0,
                confidence: 0.0,
                age: 0,
                visibility: 0.0,
            }],
        }];
        assert!(matches!(
            VideoFrameInner::try_from(&pb),
            Err(Error::UnknownTrackState(42))
        ));
    }

    #[test]
    fn test_misc_track_unknown_category() {
        let mut pb = generated::VideoFrame::from(&gen_frame());
        pb.misc_tracks = vec![generated::MiscTrackData {
            class_id: 1,
            object_id: 1,
            label: None,
            source_id: "s".into(),
            category: 7,
            frames: vec![],
        }];
        assert!(matches!(
            VideoFrameInner::try_from(&pb),
            Err(Error::UnknownMiscTrackCategory(7))
        ));
    }
}
