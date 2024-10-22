use crate::primitives::bbox::RBBox;
use crate::with_gil;
use pyo3::prelude::*;

#[pyfunction]
pub fn solely_owned_areas(bboxes: Vec<RBBox>, parallel: bool) -> Vec<f64> {
    let boxes = bboxes.iter().map(|b| &b.0).collect::<Vec<_>>();
    with_gil!(|_| { savant_core::primitives::bbox::utils::solely_owned_areas(&boxes, parallel) })
}
