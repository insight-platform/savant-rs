//! PyO3 wrappers for nvinfer Roi and RoiKind.

use crate::primitives::bbox::RBBox;
use nvinfer::{Roi, RoiKind};
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

/// Per-slot ROI specification.
///
/// Use ``RoiKind.full_frame()`` when the entire frame should be inferred,
/// or ``RoiKind.rois(list_of_roi)`` when specific regions are provided.
#[pyclass(name = "RoiKind", module = "savant_rs.nvinfer", skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct PyRoiKind(pub(crate) RoiKind);

#[pymethods]
impl PyRoiKind {
    /// Create a ``FullFrame`` variant — infer on the whole frame.
    #[staticmethod]
    fn full_frame() -> Self {
        Self(RoiKind::FullFrame)
    }

    /// Create a ``Rois`` variant with a list of [`Roi`].
    #[staticmethod]
    fn rois(rois: Vec<PyRef<'_, PyRoi>>) -> Self {
        Self(RoiKind::Rois(rois.iter().map(|r| r.to_rust()).collect()))
    }

    /// ``True`` when this is the ``FullFrame`` variant.
    #[getter]
    fn is_full_frame(&self) -> bool {
        matches!(self.0, RoiKind::FullFrame)
    }

    /// Return the ROI list (empty for ``FullFrame``).
    fn get_rois(&self) -> Vec<PyRoi> {
        match &self.0 {
            RoiKind::FullFrame => vec![],
            RoiKind::Rois(v) => v.iter().map(|r| PyRoi(r.clone())).collect(),
        }
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            RoiKind::FullFrame => "RoiKind.FullFrame".to_string(),
            RoiKind::Rois(v) => format!("RoiKind.Rois(len={})", v.len()),
        }
    }
}

impl PyRoiKind {
    pub(crate) fn to_rust(&self) -> RoiKind {
        self.0.clone()
    }
}
