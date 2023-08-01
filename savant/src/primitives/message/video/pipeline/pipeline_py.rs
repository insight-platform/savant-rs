use super::VideoPipeline;
use crate::primitives::message::video::pipeline::VideoPipelineStagePayloadType;
use crate::utils::python::release_gil;
use lazy_static::lazy_static;
use parking_lot::{const_mutex, Mutex};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

lazy_static! {
    static ref PIPELINE: Mutex<VideoPipeline> = const_mutex(VideoPipeline::default());
}

// #[pyfunction]
// #[pyo3(name = "add_stage")]
// pub fn add_stage_gil(name: String, stage_type: VideoPipelineStagePayloadType) -> PyResult<()> {
//     release_gil(|| PIPELINE.lock().add_stage(&name, stage_type))
//         .map_err(|e| PyValueError::new_err(e.to_string()))
// }
