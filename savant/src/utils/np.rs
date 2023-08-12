use nalgebra::Scalar;
use std::fmt::Debug;

pub mod np_nalgebra;
pub mod np_ndarray;

pub trait ConvF32 {
    fn conv_f32(self) -> f32;
}

pub trait RConvF32 {
    fn conv_from_f32(f: f32) -> Self;
}

pub trait ElementType: numpy::Element + Scalar + Copy + Clone + Debug {}

impl ElementType for f32 {}

impl ElementType for f64 {}

impl ElementType for i8 {}

impl ElementType for i16 {}

impl ElementType for i32 {}

impl ElementType for i64 {}

impl ElementType for u8 {}

impl ElementType for u16 {}

impl ElementType for u32 {}

impl ElementType for u64 {}
