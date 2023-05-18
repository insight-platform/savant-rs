pub mod attribute;
pub mod bbox;
pub mod message;
pub mod point;
pub mod polygonal_area;
pub mod segment;
pub mod to_json_value;

use crate::primitives::message::video::frame::PyFrameTransformation;
pub use attribute::Attribute;
pub use attribute::AttributeBuilder;
pub use attribute::Value;
pub use bbox::BBox;
pub use bbox::RBBox;
pub use message::eos::EndOfStream;
pub use message::loader::load_message;
pub use message::saver::save_message;
pub use message::video::batch::VideoFrameBatch;
pub use message::video::frame::PyVideoFrameContent;
pub use message::video::frame::VideoFrame;
pub use message::video::frame::VideoTranscodingMethod;
pub use message::video::object::Modification;
pub use message::video::object::Object;
pub use message::video::object::ParentObject;
pub use message::Message;
pub use point::Point;
pub use polygonal_area::PolygonalArea;
use pyo3::prelude::PyModule;
use pyo3::{pymodule, PyResult, Python};
pub use segment::Intersection;
pub use segment::IntersectionKind;
pub use segment::Segment;

#[pymodule]
pub fn primitives(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Point>()?;
    m.add_class::<Segment>()?;
    m.add_class::<IntersectionKind>()?;
    m.add_class::<Intersection>()?;
    m.add_class::<PolygonalArea>()?;
    m.add_class::<RBBox>()?;
    m.add_class::<BBox>()?;
    m.add_class::<Attribute>()?;
    m.add_class::<Value>()?;
    m.add_class::<Object>()?;
    m.add_class::<ParentObject>()?;
    m.add_class::<VideoFrame>()?;
    m.add_class::<VideoFrameBatch>()?;
    m.add_class::<EndOfStream>()?;
    m.add_class::<Message>()?;
    m.add_class::<VideoTranscodingMethod>()?;
    m.add_class::<PyVideoFrameContent>()?;
    m.add_class::<PyFrameTransformation>()?;
    m.add_class::<Modification>()?;
    Ok(())
}
