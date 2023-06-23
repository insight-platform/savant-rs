/// Attribute module specifies attribute code for [crate::primitives::VideoObjectProxy] and [crate::primitives::VideoFrameProxy].
///
pub mod attribute;
/// Here are decleared bounding boxes
///
pub mod bbox;
/// The draw specification used to draw objects on the frame when they are visualized.
pub mod draw;
pub mod eval_context;
/// The protocol message wrapping various objects to serialize an deserialize them.
pub mod message;
/// Simple point structure.
pub mod point;
/// A structure representing polygonal areas and functions.
pub mod polygonal_area;
pub mod proxy;
/// A line consisting of two points.
pub mod segment;
/// A trait to serialize various objects to json.
pub mod to_json_value;

use crate::primitives::message::video::frame::VideoFrameTransformationProxy;
pub use crate::primitives::message::video::object::objects_view::{
    VideoObjectBBoxKind, VideoObjectsView,
};
pub use crate::primitives::message::video::object::VideoObjectTrackingData;
use crate::primitives::proxy::video_object_rbbox::VideoObjectRBBoxProxy;
pub use attribute::attribute_value::{
    AttributeValue, AttributeValueType, AttributeValueVariant, AttributeValuesView,
};
pub use attribute::{Attribute, AttributeBuilder};
pub use bbox::{PythonBBox, RBBox};
pub use draw::*;
pub use message::eos::EndOfStream;
pub use message::loader::load_message;
pub use message::saver::save_message;
pub use message::video::batch::VideoFrameBatch;
pub use message::video::frame::frame_update::VideoFrameUpdate;
pub use message::video::frame::frame_update::{
    AttributeUpdateCollisionResolutionPolicy, AttributeUpdateCollisionResolutionPolicyProxy,
    ObjectUpdateCollisionResolutionPolicy, VideoObjectUpdateCollisionResolutionPolicyProxy,
};
pub use message::video::frame::{
    VideoFrameContentProxy, VideoFrameProxy, VideoFrameTranscodingMethod,
};
pub use message::video::object::{VideoObjectModification, VideoObjectProxy};
pub use message::Message;
pub use point::Point;
pub use polygonal_area::PolygonalArea;
use pyo3::prelude::PyModule;
use pyo3::{pymodule, PyResult, Python};
pub use segment::{Intersection, IntersectionKind, Segment};

#[pymodule]
pub fn geometry(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Point>()?;
    m.add_class::<Segment>()?;
    m.add_class::<IntersectionKind>()?;
    m.add_class::<Intersection>()?;
    m.add_class::<PolygonalArea>()?;
    m.add_class::<RBBox>()?;
    m.add_class::<PythonBBox>()?;
    Ok(())
}

#[pymodule]
pub fn draw_spec(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<ColorDraw>()?;
    m.add_class::<BoundingBoxDraw>()?;
    m.add_class::<DotDraw>()?;
    m.add_class::<LabelDraw>()?;
    m.add_class::<LabelPositionKind>()?;
    m.add_class::<LabelPosition>()?;
    m.add_class::<PaddingDraw>()?;
    m.add_class::<ObjectDraw>()?;
    m.add_class::<SetDrawLabelKindProxy>()?;
    Ok(())
}

#[pymodule]
pub fn primitives(_py: Python, m: &PyModule) -> PyResult<()> {
    use AttributeUpdateCollisionResolutionPolicyProxy as AttributeUpdateCollisionResolutionPolicy;
    use VideoFrameContentProxy as VideoFrameContent;
    use VideoFrameTransformationProxy as VideoFrameTransformation;
    use VideoObjectUpdateCollisionResolutionPolicyProxy as VideoObjectUpdateCollisionResolutionPolicy;

    m.add_class::<Attribute>()?;
    m.add_class::<AttributeUpdateCollisionResolutionPolicy>()?;
    m.add_class::<VideoObjectUpdateCollisionResolutionPolicy>()?;
    m.add_class::<AttributeValue>()?;
    m.add_class::<AttributeValueType>()?;
    m.add_class::<AttributeValuesView>()?;
    m.add_class::<EndOfStream>()?;

    m.add_class::<VideoFrameProxy>()?;
    m.add_class::<VideoFrameBatch>()?;
    m.add_class::<VideoFrameContent>()?;
    m.add_class::<VideoFrameTranscodingMethod>()?;
    m.add_class::<VideoFrameUpdate>()?;
    m.add_class::<VideoFrameTransformation>()?;

    m.add_class::<VideoObjectProxy>()?;
    m.add_class::<VideoObjectModification>()?;
    m.add_class::<VideoObjectTrackingData>()?;
    m.add_class::<VideoObjectRBBoxProxy>()?;
    m.add_class::<VideoObjectsView>()?;

    Ok(())
}
