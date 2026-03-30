//! Python wrappers for geometry and transform configuration types.

use super::enums::{PyComputeMode, PyInterpolation, PyPadding};
use deepstream_buffers::transform::Rect;
use deepstream_buffers::{DstPadding, TransformConfig};
use pyo3::prelude::*;

// ─── Rect ────────────────────────────────────────────────────────────────

/// A rectangle in pixel coordinates (top, left, width, height).
///
/// Used as an optional source crop region for transform and send_frame.
#[pyclass(name = "Rect", module = "savant_rs.deepstream", skip_from_py_object)]
#[derive(Debug, Clone, Copy)]
pub struct PyRect {
    #[pyo3(get, set)]
    pub top: u32,
    #[pyo3(get, set)]
    pub left: u32,
    #[pyo3(get, set)]
    pub width: u32,
    #[pyo3(get, set)]
    pub height: u32,
}

#[pymethods]
impl PyRect {
    #[new]
    #[pyo3(signature = (top, left, width, height))]
    fn new(top: u32, left: u32, width: u32, height: u32) -> Self {
        Self {
            top,
            left,
            width,
            height,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Rect(top={}, left={}, width={}, height={})",
            self.top, self.left, self.width, self.height
        )
    }
}

impl PyRect {
    pub(crate) fn into_rust(self) -> Rect {
        Rect {
            top: self.top,
            left: self.left,
            width: self.width,
            height: self.height,
        }
    }
}

// ─── DstPadding ──────────────────────────────────────────────────────────

/// Optional per-side destination padding for letterboxing.
///
/// When set in ``TransformConfig.dst_padding``, reduces the effective
/// destination area before the letterbox rect is computed.
#[pyclass(from_py_object, name = "DstPadding", module = "savant_rs.deepstream")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PyDstPadding {
    #[pyo3(get, set)]
    pub left: u32,
    #[pyo3(get, set)]
    pub top: u32,
    #[pyo3(get, set)]
    pub right: u32,
    #[pyo3(get, set)]
    pub bottom: u32,
}

#[pymethods]
impl PyDstPadding {
    #[new]
    #[pyo3(signature = (left = 0, top = 0, right = 0, bottom = 0))]
    fn new(left: u32, top: u32, right: u32, bottom: u32) -> Self {
        Self {
            left,
            top,
            right,
            bottom,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "DstPadding(left={}, top={}, right={}, bottom={})",
            self.left, self.top, self.right, self.bottom
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }

    fn __hash__(&self) -> u64 {
        let mut h = self.left as u64;
        h = h.wrapping_mul(31).wrapping_add(self.top as u64);
        h = h.wrapping_mul(31).wrapping_add(self.right as u64);
        h = h.wrapping_mul(31).wrapping_add(self.bottom as u64);
        h
    }

    /// Create destination padding with equal values on all sides.
    ///
    /// Args:
    ///     value: Padding value applied to left, top, right, and bottom.
    ///
    /// Returns:
    ///     A new ``DstPadding`` with all sides set to *value*.
    #[staticmethod]
    fn uniform(value: u32) -> Self {
        Self {
            left: value,
            top: value,
            right: value,
            bottom: value,
        }
    }
}

impl From<PyDstPadding> for DstPadding {
    fn from(p: PyDstPadding) -> Self {
        DstPadding {
            left: p.left,
            top: p.top,
            right: p.right,
            bottom: p.bottom,
        }
    }
}

// ─── TransformConfig ────────────────────────────────────────────────────

/// Configuration for a transform (scale / letterbox) operation.
///
/// All fields have sensible defaults (``Padding.SYMMETRIC``,
/// ``Interpolation.BILINEAR``, ``ComputeMode.DEFAULT``).
#[pyclass(
    from_py_object,
    name = "TransformConfig",
    module = "savant_rs.deepstream"
)]
#[derive(Debug, Clone)]
pub struct PyTransformConfig {
    #[pyo3(get, set)]
    pub padding: PyPadding,
    #[pyo3(get, set)]
    pub dst_padding: Option<PyDstPadding>,
    #[pyo3(get, set)]
    pub interpolation: PyInterpolation,
    #[pyo3(get, set)]
    pub compute_mode: PyComputeMode,
}

#[pymethods]
impl PyTransformConfig {
    #[new]
    #[pyo3(signature = (
        padding = PyPadding::Symmetric,
        dst_padding = None,
        interpolation = PyInterpolation::Bilinear,
        compute_mode = PyComputeMode::Default,
    ))]
    fn new(
        padding: PyPadding,
        dst_padding: Option<PyDstPadding>,
        interpolation: PyInterpolation,
        compute_mode: PyComputeMode,
    ) -> Self {
        Self {
            padding,
            dst_padding,
            interpolation,
            compute_mode,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "TransformConfig(padding={:?}, dst_padding={:?}, interpolation={:?}, compute_mode={:?})",
            self.padding, self.dst_padding, self.interpolation, self.compute_mode,
        )
    }
}

impl PyTransformConfig {
    pub(crate) fn to_rust(&self) -> TransformConfig {
        TransformConfig {
            padding: self.padding.into(),
            dst_padding: self.dst_padding.map(Into::into),
            interpolation: self.interpolation.into(),
            compute_mode: self.compute_mode.into(),
            cuda_stream: deepstream_buffers::CudaStream::default(),
        }
    }
}
