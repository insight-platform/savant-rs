use crate::detach;
use crate::primitives::bbox::{BBoxMetricType, RBBox};
use pyo3::prelude::*;
use std::collections::HashMap;

#[pyfunction]
pub fn solely_owned_areas(bboxes: Vec<RBBox>, parallel: bool) -> Vec<f64> {
    let boxes = bboxes.iter().map(|b| &b.0).collect::<Vec<_>>();
    detach!(true, || {
        savant_core::primitives::bbox::utils::solely_owned_areas(&boxes, parallel)
    })
}

#[pyfunction]
pub fn associate_bboxes(
    candidates: Vec<RBBox>,
    owners: Vec<RBBox>,
    metric: BBoxMetricType,
    threshold: f32,
) -> HashMap<usize, Vec<(usize, f32)>> {
    let candidates = candidates.iter().map(|b| &b.0).collect::<Vec<_>>();
    let owners = owners.iter().map(|b| &b.0).collect::<Vec<_>>();
    detach!(true, || {
        savant_core::primitives::bbox::utils::associate_bboxes(
            &candidates,
            &owners,
            metric.into(),
            threshold,
        )
    })
}
