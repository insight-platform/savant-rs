pub mod attribute;
pub use attribute::*;
pub mod point;
pub use point::*;
pub mod polygonal_area;
pub use polygonal_area::*;
pub mod bbox;
pub use bbox::*;

pub mod any_object;
pub mod attribute_set;
pub mod attribute_value;
pub mod eos;
pub mod frame;
pub mod frame_batch;
pub mod frame_update;
pub mod gstreamer_frame_time;
pub mod misc_track;
pub mod object;
pub mod segment;
pub mod shutdown;
pub mod userdata;
pub mod video_codec;

pub use segment::*;

pub mod rust {
    pub use super::attribute::Attribute;
    pub use super::attribute_set::AttributeSet;
    pub use super::attribute_value::AttributeValue;
    pub use super::bbox::BBoxMetricType;
    pub use super::bbox::RBBox;
    pub use super::bbox::RBBoxData;
    pub use super::eos::EndOfStream;
    pub use super::frame::BelongingVideoFrame;
    pub use super::frame::VideoFrameContent;
    pub use super::frame::VideoFrameProxy;
    pub use super::frame::VideoFrameTranscodingMethod;
    pub use super::frame::VideoFrameTransformation;
    pub use super::frame::VideoObjectTree;
    pub use super::frame_batch::VideoFrameBatch;
    pub use super::frame_update::VideoFrameUpdate;
    pub use super::gstreamer_frame_time::{
        frame_clock_ns, normalize_frame_to_gst_ns, time_base_to_ns, FrameClockNs, GST_TIME_BASE,
    };
    pub use super::misc_track::MiscTrackCategory;
    pub use super::misc_track::MiscTrackData;
    pub use super::misc_track::MiscTrackFrame;
    pub use super::misc_track::TrackState;
    pub use super::misc_track::TrackUpdate;
    pub use super::object::BorrowedVideoObject;
    pub use super::object::VideoObject;
    pub use super::object::VideoObjectBBoxTransformation;
    pub use super::object::VideoObjectBuilder;
    pub use super::point::Point;
    pub use super::polygonal_area::PolygonalArea;
    pub use super::segment::Intersection;
    pub use super::segment::IntersectionKind;
    pub use super::segment::Segment;
    pub use super::shutdown::Shutdown;
    pub use super::userdata::UserData;
    pub use super::video_codec::VideoCodec;
    pub use crate::message::Message;
    pub use crate::primitives::frame::ExternalFrame;
    pub use crate::primitives::object::IdCollisionResolutionPolicy;
}
