pub mod attribute;
pub mod bbox;
pub mod draw;
pub mod message;
pub mod point;
pub mod polygonal_area;
pub mod segment;
pub mod to_json_value;

use crate::primitives::message::video::frame::PyFrameTransformation;
use crate::primitives::message::video::object::vector::VectorView;
use crate::primitives::message::video::object::ObjectTrack;
pub use attribute::Attribute;
pub use attribute::AttributeBuilder;
pub use attribute::Value;
pub use bbox::{PythonBBox, RBBox};
pub use draw::*;
pub use message::eos::EndOfStream;
pub use message::loader::load_message;
pub use message::saver::save_message;
pub use message::video::batch::VideoFrameBatch;
pub use message::video::frame::{PyVideoFrameContent, VideoFrame, VideoTranscodingMethod};
pub use message::video::object::{Modification, Object};
pub use message::Message;
pub use point::Point;
pub use polygonal_area::PolygonalArea;
use pyo3::prelude::PyModule;
use pyo3::{pymodule, PyResult, Python};
pub use segment::{Intersection, IntersectionKind, Segment};

#[pymodule]
pub fn primitives(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Point>()?;
    m.add_class::<Segment>()?;
    m.add_class::<IntersectionKind>()?;
    m.add_class::<Intersection>()?;
    m.add_class::<PolygonalArea>()?;
    m.add_class::<RBBox>()?;
    m.add_class::<PythonBBox>()?;
    m.add_class::<Attribute>()?;
    m.add_class::<Value>()?;
    m.add_class::<Object>()?;
    m.add_class::<ObjectTrack>()?;
    m.add_class::<VectorView>()?;
    m.add_class::<VideoFrame>()?;
    m.add_class::<VideoFrameBatch>()?;
    m.add_class::<EndOfStream>()?;
    m.add_class::<Message>()?;
    m.add_class::<VideoTranscodingMethod>()?;
    m.add_class::<PyVideoFrameContent>()?;
    m.add_class::<PyFrameTransformation>()?;
    m.add_class::<Modification>()?;

    m.add_class::<ColorDraw>()?;
    m.add_class::<BoundingBoxDraw>()?;
    m.add_class::<DotDraw>()?;
    m.add_class::<LabelDraw>()?;
    m.add_class::<LabelPositionKind>()?;
    m.add_class::<LabelPosition>()?;
    m.add_class::<PaddingDraw>()?;
    m.add_class::<ObjectDraw>()?;
    m.add_class::<SetDrawLabelKindWrapper>()?;
    Ok(())
}
