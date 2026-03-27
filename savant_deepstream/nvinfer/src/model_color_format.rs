//! Model input color format for the nvinfer config.

/// Color space the model expects for its input tensor.
///
/// Maps to nvinfer's `model-color-format` property:
///
/// | Variant | nvinfer value | Channels |
/// |---------|---------------|----------|
/// | [`RGB`](Self::RGB)   | `0` | 3 |
/// | [`BGR`](Self::BGR)   | `1` | 3 |
/// | [`GRAY`](Self::GRAY) | `2` | 1 |
///
/// This key is auto-injected by
/// [`NvInferConfig::validate_and_materialize`](crate::config::NvInferConfig::validate_and_materialize)
/// and must not appear in `nvinfer_properties`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModelColorFormat {
    /// 3-channel RGB input (`model-color-format=0`).
    #[default]
    RGB,
    /// 3-channel BGR input (`model-color-format=1`).
    BGR,
    /// Single-channel grayscale input (`model-color-format=2`).
    GRAY,
}

impl ModelColorFormat {
    /// Number of channels the model expects.
    pub fn channels(&self) -> u32 {
        match self {
            Self::RGB | Self::BGR => 3,
            Self::GRAY => 1,
        }
    }

    /// The numeric string value for the nvinfer config file.
    pub fn nvinfer_value(&self) -> &'static str {
        match self {
            Self::RGB => "0",
            Self::BGR => "1",
            Self::GRAY => "2",
        }
    }
}

impl std::fmt::Display for ModelColorFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RGB => f.write_str("RGB"),
            Self::BGR => f.write_str("BGR"),
            Self::GRAY => f.write_str("GRAY"),
        }
    }
}
