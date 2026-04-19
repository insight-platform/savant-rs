use crate::deepstream::encoder_config::PyEncoderConfig;
use crate::deepstream::PyTransformConfig;
use picasso::prelude::CodecSpec;
use pyo3::prelude::*;

/// Describes what to do with each incoming frame for a given source.
///
/// This is a tagged union exposed via factory static methods:
///
/// - `CodecSpec.drop_frames()` -- discard frames entirely.
/// - `CodecSpec.bypass()` -- pass frames through without encoding.
/// - `CodecSpec.encode(transform, encoder)` -- transform + optional render + encode.
#[pyclass(from_py_object, name = "CodecSpec", module = "savant_rs.picasso")]
#[derive(Debug, Clone)]
pub struct PyCodecSpec {
    inner: CodecSpec,
}

#[pymethods]
impl PyCodecSpec {
    /// Discard the frame entirely.
    #[staticmethod]
    fn drop_frames() -> Self {
        Self {
            inner: CodecSpec::Drop,
        }
    }

    /// Pass the frame through without encoding -- only transform bboxes back
    /// to initial coordinates.
    #[staticmethod]
    fn bypass() -> Self {
        Self {
            inner: CodecSpec::Bypass,
        }
    }

    /// GPU-transform the frame to a target resolution, optionally render Skia
    /// overlays, then encode.
    ///
    /// Raises ``ValueError`` when the supplied [`PyEncoderConfig`] carries
    /// ``encoder_params`` whose codec or build-platform variant does not
    /// match the configured codec / current build target.
    #[staticmethod]
    fn encode(transform: &PyTransformConfig, encoder: &PyEncoderConfig) -> PyResult<Self> {
        let transform_cfg = transform.to_rust();
        let encoder_cfg = encoder.to_rust()?;
        Ok(Self {
            inner: CodecSpec::Encode {
                transform: transform_cfg,
                encoder: Box::new(encoder_cfg),
            },
        })
    }

    /// `True` when this spec drops frames.
    #[getter]
    fn is_drop(&self) -> bool {
        matches!(self.inner, CodecSpec::Drop)
    }

    /// `True` when this spec bypasses encoding.
    #[getter]
    fn is_bypass(&self) -> bool {
        matches!(self.inner, CodecSpec::Bypass)
    }

    /// `True` when this spec encodes frames.
    #[getter]
    fn is_encode(&self) -> bool {
        matches!(self.inner, CodecSpec::Encode { .. })
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            CodecSpec::Drop => "CodecSpec.drop_frames()".to_string(),
            CodecSpec::Bypass => "CodecSpec.bypass()".to_string(),
            CodecSpec::Encode {
                transform, encoder, ..
            } => {
                format!("CodecSpec.encode(transform={transform:?}, encoder={encoder:?})")
            }
        }
    }
}

impl PyCodecSpec {
    pub(crate) fn to_rust(&self) -> CodecSpec {
        self.inner.clone()
    }

    pub(crate) fn default_drop() -> Self {
        Self {
            inner: CodecSpec::Drop,
        }
    }
}
