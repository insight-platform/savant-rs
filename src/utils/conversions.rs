use crate::utils::np::{ConvF64, RConvF64};

impl ConvF64 for f32 {
    fn conv_f64(self) -> f64 {
        self as f64
    }
}

impl ConvF64 for f64 {
    fn conv_f64(self) -> f64 {
        self
    }
}

impl ConvF64 for i32 {
    fn conv_f64(self) -> f64 {
        self as f64
    }
}

impl ConvF64 for i64 {
    fn conv_f64(self) -> f64 {
        self as f64
    }
}

impl RConvF64 for f32 {
    fn conv_from_f64(f: f64) -> Self {
        f as f32
    }
}

impl RConvF64 for f64 {
    fn conv_from_f64(f: f64) -> Self {
        f
    }
}

impl RConvF64 for i32 {
    fn conv_from_f64(f: f64) -> Self {
        f as i32
    }
}

impl RConvF64 for i64 {
    fn conv_from_f64(f: f64) -> Self {
        f as i64
    }
}
