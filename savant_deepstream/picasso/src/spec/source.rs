use super::codec::CodecSpec;
use super::conditional::ConditionalSpec;
use super::draw::ObjectDrawSpec;

/// Controls when the `on_gpumat` callback fires relative to Skia rendering.
///
/// - [`SkiaGpuMat`](Self::SkiaGpuMat) — Skia draws first, then `on_gpumat`.
/// - [`GpuMatSkia`](Self::GpuMatSkia) — `on_gpumat` fires first, then Skia.
/// - [`GpuMatSkiaGpuMat`](Self::GpuMatSkiaGpuMat) — `on_gpumat` fires before
///   **and** after Skia rendering.
///
/// After each `on_gpumat` invocation the worker's CUDA stream is
/// synchronised before the next pipeline stage proceeds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CallbackInvocationOrder {
    /// Skia render → `on_gpumat` (default, preserves legacy behaviour).
    #[default]
    SkiaGpuMat,
    /// `on_gpumat` → Skia render.
    GpuMatSkia,
    /// `on_gpumat` → Skia render → `on_gpumat`.
    GpuMatSkiaGpuMat,
}

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
    /// When to fire `on_gpumat` relative to Skia rendering.
    pub callback_order: CallbackInvocationOrder,
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
            callback_order: CallbackInvocationOrder::default(),
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

    #[test]
    fn default_callback_order() {
        assert_eq!(
            CallbackInvocationOrder::default(),
            CallbackInvocationOrder::SkiaGpuMat,
        );
    }

    #[test]
    fn source_spec_default_has_skia_gpumat() {
        let s = SourceSpec::default();
        assert_eq!(s.callback_order, CallbackInvocationOrder::SkiaGpuMat);
    }
}
