use crate::utils::np::{ConvF32, RConvF32};

impl ConvF32 for f32 {
    fn conv_f32(self) -> f32 {
        self
    }
}

impl ConvF32 for f64 {
    fn conv_f32(self) -> f32 {
        self as f32
    }
}

impl ConvF32 for i32 {
    fn conv_f32(self) -> f32 {
        self as f32
    }
}

impl ConvF32 for i64 {
    fn conv_f32(self) -> f32 {
        self as f32
    }
}

impl RConvF32 for f32 {
    fn conv_from_f32(f: f32) -> Self {
        f
    }
}

impl RConvF32 for f64 {
    fn conv_from_f32(f: f32) -> Self {
        f as f64
    }
}

impl RConvF32 for i32 {
    fn conv_from_f32(f: f32) -> Self {
        f as i32
    }
}

impl RConvF32 for i64 {
    fn conv_from_f32(f: f32) -> Self {
        f as i64
    }
}
