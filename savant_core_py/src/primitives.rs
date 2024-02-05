/// Attribute module specifies attribute code for [crate::primitives::BorrowedVideoObject] and [crate::primitives::VideoFrame].
///
pub mod attribute;
pub mod attribute_value;
pub mod batch;
/// Here are decleared bounding boxes
///
pub mod bbox;
pub mod eos;
pub mod frame;
pub mod frame_update;
pub mod message;
pub mod object;
pub mod objects_view;
/// Simple point structure.
pub mod point;
/// A structure representing polygonal areas and functions.
pub mod polygonal_area;
/// Implementation for Python attributes in VideoObject and VideoFrame.
pub mod pyobject;
/// A line consisting of two points.
pub mod segment;
pub mod shutdown;
pub mod user_data;

use crate::primitives::frame::{
    VideoFrame, VideoFrameContent, VideoFrameTranscodingMethod, VideoFrameTransformation,
};

use crate::primitives::attribute::Attribute;
use crate::primitives::attribute_value::{AttributeValue, AttributeValueType, AttributeValuesView};
use crate::primitives::batch::VideoFrameBatch;
use crate::primitives::bbox::{BBox, RBBox};

use crate::primitives::eos::EndOfStream;
use crate::primitives::frame_update::{
    AttributeUpdatePolicy, ObjectUpdatePolicy, VideoFrameUpdate,
};
use crate::primitives::object::{BorrowedVideoObject, IdCollisionResolutionPolicy, VideoObject};
use crate::primitives::objects_view::VideoObjectsView;
use crate::primitives::point::Point;
use crate::primitives::polygonal_area::PolygonalArea;
use crate::primitives::segment::{Intersection, IntersectionKind, Segment};
use crate::primitives::shutdown::Shutdown;
use crate::primitives::user_data::UserData;
use pyo3::prelude::PyModule;
use pyo3::{pymodule, PyResult, Python};