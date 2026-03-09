//! PyO3 wrapper for nvinfer Roi.

use crate::deepstream::PyRect;
use nvinfer::Roi;
use pyo3::prelude::*;

/// A region of interest: an identifier paired with a bounding box.
///
/// Args:
///     id (int): Caller-defined identifier returned in ``ElementOutput.roi_id``.
///     rect (Rect): Bounding box (top, left, width, height) in pixel coordinates.
#[pyclass(name = "Roi", module = "savant_rs.nvinfer", skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct PyRoi {
    inner_id: i64,
    inner_rect: PyRect,
}

#[pymethods]
impl PyRoi {
    #[new]
    #[pyo3(signature = (id, rect))]
    fn new(id: i64, rect: PyRef<'_, PyRect>) -> Self {
        Self {
            inner_id: id,
            inner_rect: *rect,
        }
    }

    /// Caller-defined identifier.
    #[getter]
    fn id(&self) -> i64 {
        self.inner_id
    }

    /// Bounding box in pixel coordinates.
    #[getter]
    fn rect(&self) -> PyRect {
        self.inner_rect
    }

    fn __repr__(&self) -> String {
        format!(
            "Roi(id={}, rect=Rect(top={}, left={}, width={}, height={}))",
            self.inner_id,
            self.inner_rect.top,
            self.inner_rect.left,
            self.inner_rect.width,
            self.inner_rect.height
        )
    }
}

impl PyRoi {
    pub(crate) fn to_rust(&self) -> Roi {
        Roi {
            id: self.inner_id,
            rect: self.inner_rect.into_rust(),
        }
    }
}
