//! PyO3 wrapper for nvinfer Roi.

use crate::primitives::bbox::RBBox;
use nvinfer::Roi;
use pyo3::prelude::*;

/// A region of interest: an identifier paired with a bounding box.
///
/// Args:
///     id (int): Caller-defined identifier returned in ``ElementOutput.roi_id``.
///     bbox (RBBox): Bounding box (center-based, optionally rotated).
#[pyclass(name = "Roi", module = "savant_rs.nvinfer", skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct PyRoi(Roi);

#[pymethods]
impl PyRoi {
    #[new]
    #[pyo3(signature = (id, bbox))]
    fn new(id: i64, bbox: &RBBox) -> Self {
        Self(Roi {
            id,
            bbox: bbox.0.clone(),
        })
    }

    /// Caller-defined identifier.
    #[getter]
    fn id(&self) -> i64 {
        self.0.id
    }

    /// Bounding box (center-based, optionally rotated).
    #[getter]
    fn bbox(&self) -> RBBox {
        RBBox(self.0.bbox.clone())
    }

    fn __repr__(&self) -> String {
        let b = &self.0.bbox;
        format!(
            "Roi(id={}, bbox=RBBox(xc={}, yc={}, w={}, h={}, angle={:?}))",
            self.0.id,
            b.get_xc(),
            b.get_yc(),
            b.get_width(),
            b.get_height(),
            b.get_angle(),
        )
    }
}

impl PyRoi {
    pub(crate) fn to_rust(&self) -> Roi {
        self.0.clone()
    }
}
