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

use crate::primitives::frame::VideoFrame;

use crate::primitives::attribute::Attribute;

use crate::primitives::batch::VideoFrameBatch;
use crate::primitives::bbox::RBBox;

use crate::primitives::eos::EndOfStream;

use crate::primitives::point::Point;
use crate::primitives::polygonal_area::PolygonalArea;
use crate::primitives::segment::{Intersection, Segment};
use crate::primitives::shutdown::Shutdown;
