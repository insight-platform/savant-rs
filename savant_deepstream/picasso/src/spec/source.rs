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

impl SourceSpec {
    /// Create a new builder starting from default values.
    pub fn builder() -> SourceSpecBuilder {
        SourceSpecBuilder(SourceSpec::default())
    }
}

/// Builder for [`SourceSpec`] — uses `Default` values for unset fields.
///
/// # Example
///
/// ```rust,ignore
/// let spec = SourceSpec::builder()
///     .font_family("monospace")
///     .use_on_render(true)
///     .build();
/// ```
pub struct SourceSpecBuilder(SourceSpec);

impl SourceSpecBuilder {
    pub fn codec(mut self, codec: CodecSpec) -> Self {
        self.0.codec = codec;
        self
    }

    pub fn conditional(mut self, conditional: ConditionalSpec) -> Self {
        self.0.conditional = conditional;
        self
    }

    pub fn draw(mut self, draw: ObjectDrawSpec) -> Self {
        self.0.draw = draw;
        self
    }

    pub fn font_family(mut self, font_family: impl Into<String>) -> Self {
        self.0.font_family = font_family.into();
        self
    }

    pub fn idle_timeout_secs(mut self, secs: u64) -> Self {
        self.0.idle_timeout_secs = Some(secs);
        self
    }

    pub fn use_on_render(mut self, enable: bool) -> Self {
        self.0.use_on_render = enable;
        self
    }

    pub fn use_on_gpumat(mut self, enable: bool) -> Self {
        self.0.use_on_gpumat = enable;
        self
    }

    pub fn callback_order(mut self, order: CallbackInvocationOrder) -> Self {
        self.0.callback_order = order;
        self
    }

    /// Finish building and return the [`SourceSpec`].
    pub fn build(self) -> SourceSpec {
        self.0
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

    #[test]
    fn builder_default_matches_struct_default() {
        let built = SourceSpec::builder().build();
        let direct = SourceSpec::default();
        assert!(matches!(built.codec, CodecSpec::Drop));
        assert_eq!(built.font_family, direct.font_family);
        assert_eq!(built.use_on_render, direct.use_on_render);
    }

    #[test]
    fn builder_overrides_fields() {
        let spec = SourceSpec::builder()
            .font_family("monospace")
            .use_on_render(true)
            .idle_timeout_secs(60)
            .build();
        assert_eq!(spec.font_family, "monospace");
        assert!(spec.use_on_render);
        assert_eq!(spec.idle_timeout_secs, Some(60));
    }
}
