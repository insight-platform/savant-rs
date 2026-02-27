use super::codec::CodecSpec;
use super::conditional::ConditionalSpec;
use super::draw::ObjectDrawSpec;

/// Complete per-source configuration combining all spec facets.
#[derive(Debug, Clone)]
pub struct SourceSpec {
    /// What to do with frames (drop / bypass / encode / render+encode).
    pub codec: CodecSpec,
    /// Attribute gates for conditional processing.
    pub conditional: ConditionalSpec,
    /// Static draw specs for object overlays.
    pub draw: ObjectDrawSpec,
    /// Font family name used by the internal object drawing pipeline
    /// (bounding-box labels).  Defaults to `"sans-serif"`.
    pub font_family: String,
    /// Per-source idle timeout override (seconds).
    /// `None` means use the engine's `GeneralSpec::idle_timeout_secs`.
    pub idle_timeout_secs: Option<u64>,
    /// Whether to fire the `on_render` callback.
    pub use_on_render: bool,
    /// Whether to fire the `on_gpumat` callback.
    pub use_on_gpumat: bool,
}

impl Default for SourceSpec {
    fn default() -> Self {
        Self {
            codec: CodecSpec::Drop,
            conditional: ConditionalSpec::default(),
            draw: ObjectDrawSpec::default(),
            font_family: "sans-serif".to_string(),
            idle_timeout_secs: None,
            use_on_render: false,
            use_on_gpumat: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_drop() {
        let s = SourceSpec::default();
        assert!(matches!(s.codec, CodecSpec::Drop));
        assert!(!s.use_on_render);
        assert!(!s.use_on_gpumat);
        assert!(s.idle_timeout_secs.is_none());
    }
}
