use crate::primitives::bbox::RBBox;
use lazy_static::lazy_static;

pub const EPS: f32 = 0.00001;
pub const BBOX_ELEMENT_UNDEFINED: f32 = 3.402_823_5e38_f32;

lazy_static! {
    pub static ref BBOX_UNDEFINED: RBBox = RBBox::new(
        BBOX_ELEMENT_UNDEFINED,
        BBOX_ELEMENT_UNDEFINED,
        BBOX_ELEMENT_UNDEFINED,
        BBOX_ELEMENT_UNDEFINED,
        None,
    );
}
