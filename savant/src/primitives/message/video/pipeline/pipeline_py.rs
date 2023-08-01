use super::VideoPipeline;
use crate::primitives::message::video::pipeline::{
    VideoPipelineStagePayloadType, VideoPipelineTelemetryMessage,
};
use crate::primitives::{VideoFrameBatch, VideoFrameProxy, VideoFrameUpdate};
use crate::utils::python::release_gil;
use lazy_static::lazy_static;
use parking_lot::{const_mutex, Mutex};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;

lazy_static! {
    static ref PIPELINE: Mutex<VideoPipeline> = const_mutex(VideoPipeline::default());
}

#[pyfunction]
#[pyo3(name = "add_stage")]
fn add_stage_gil(name: String, stage_type: VideoPipelineStagePayloadType) -> PyResult<()> {
    release_gil(|| PIPELINE.lock().add_stage(&name, stage_type))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(name = "retrieve_telemetry")]
fn retrieve_telemetry_gil() -> Vec<VideoPipelineTelemetryMessage> {
    release_gil(|| PIPELINE.lock().retrieve_telemetry())
}

#[pyfunction]
#[pyo3(name = "get_stage_type")]
fn get_stage_type_gil(name: String) -> Option<VideoPipelineStagePayloadType> {
    release_gil(|| PIPELINE.lock().get_stage_type(&name).cloned())
}

#[pyfunction]
#[pyo3(name = "add_frame_update")]
fn add_frame_update_gil(stage: String, frame_id: i64, update: VideoFrameUpdate) -> PyResult<()> {
    release_gil(|| {
        PIPELINE
            .lock()
            .add_frame_update(&stage, frame_id, update)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })
}

#[pyfunction]
#[pyo3(name = "add_batched_frame_update")]
fn add_batched_frame_update_gil(
    stage: String,
    batch_id: i64,
    frame_id: i64,
    update: VideoFrameUpdate,
) -> PyResult<()> {
    release_gil(|| {
        PIPELINE
            .lock()
            .add_batched_frame_update(&stage, batch_id, frame_id, update)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })
}

#[pyfunction]
#[pyo3(name = "add_frame")]
fn add_frame_gil(stage_name: String, frame: VideoFrameProxy) -> PyResult<i64> {
    release_gil(|| {
        PIPELINE
            .lock()
            .add_frame(&stage_name, frame)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })
}

#[pyfunction]
#[pyo3(name = "add_batch")]
fn add_batch_gil(stage_name: String, batch: VideoFrameBatch) -> PyResult<i64> {
    release_gil(|| {
        PIPELINE
            .lock()
            .add_batch(&stage_name, batch)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })
}

#[pyfunction]
#[pyo3(name = "delete")]
fn delete_gil(stage_name: String, id: i64) -> PyResult<()> {
    release_gil(|| {
        PIPELINE
            .lock()
            .delete(&stage_name, id)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })
}

#[pyfunction]
#[pyo3(name = "get_stage_len")]
fn get_stage_len_gil(stage_name: String) -> PyResult<usize> {
    release_gil(|| {
        PIPELINE
            .lock()
            .get_stage_len(&stage_name)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })
}

#[pyfunction]
#[pyo3(name = "get_independent_frame")]
fn get_independent_frame_gil(stage: String, frame_id: i64) -> PyResult<VideoFrameProxy> {
    release_gil(|| {
        PIPELINE
            .lock()
            .get_independent_frame(&stage, frame_id)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })
}

#[pyfunction]
#[pyo3(name = "get_batched_frame")]
fn get_batched_frame_gil(stage: String, batch_id: i64, frame_id: i64) -> PyResult<VideoFrameProxy> {
    release_gil(|| {
        PIPELINE
            .lock()
            .get_batched_frame(&stage, batch_id, frame_id)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })
}

#[pyfunction]
#[pyo3(name = "get_batch")]
fn get_batch_gil(stage: String, batch_id: i64) -> PyResult<VideoFrameBatch> {
    release_gil(|| {
        PIPELINE
            .lock()
            .get_batch(&stage, batch_id)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })
}

#[pyfunction]
#[pyo3(name = "apply_updates")]
fn apply_updates_gil(stage: String, id: i64) -> PyResult<()> {
    release_gil(|| {
        PIPELINE
            .lock()
            .apply_updates(&stage, id)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })
}

#[pyfunction]
#[pyo3(name = "move_as_is")]
fn move_as_is_gil(
    source_stage_name: String,
    dest_stage_name: String,
    object_ids: Vec<i64>,
) -> PyResult<()> {
    release_gil(|| {
        PIPELINE
            .lock()
            .move_as_is(&source_stage_name, &dest_stage_name, object_ids)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })
}

#[pyfunction]
#[pyo3(name = "move_and_pack_frames")]
fn move_and_pack_frames_gil(
    source_stage_name: String,
    dest_stage_name: String,
    frame_ids: Vec<i64>,
) -> PyResult<i64> {
    release_gil(|| {
        PIPELINE
            .lock()
            .move_and_pack_frames(&source_stage_name, &dest_stage_name, frame_ids)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })
}

#[pyfunction]
#[pyo3(name = "move_and_unpack_batch")]
fn move_and_unpack_batch_gil(
    source_stage_name: String,
    dest_stage_name: String,
    batch_id: i64,
) -> PyResult<HashMap<String, i64>> {
    release_gil(|| {
        PIPELINE
            .lock()
            .move_and_unpack_batch(&source_stage_name, &dest_stage_name, batch_id)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })
}

#[pymodule]
pub(crate) fn pipeline(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<VideoPipelineStagePayloadType>()?;
    m.add_class::<VideoPipelineTelemetryMessage>()?;

    m.add_function(wrap_pyfunction!(add_stage_gil, m)?)?;
    m.add_function(wrap_pyfunction!(retrieve_telemetry_gil, m)?)?;
    m.add_function(wrap_pyfunction!(get_stage_type_gil, m)?)?;
    m.add_function(wrap_pyfunction!(add_frame_update_gil, m)?)?;
    m.add_function(wrap_pyfunction!(add_batched_frame_update_gil, m)?)?;
    m.add_function(wrap_pyfunction!(add_frame_gil, m)?)?;
    m.add_function(wrap_pyfunction!(add_batch_gil, m)?)?;
    m.add_function(wrap_pyfunction!(delete_gil, m)?)?;
    m.add_function(wrap_pyfunction!(get_stage_len_gil, m)?)?;
    m.add_function(wrap_pyfunction!(get_independent_frame_gil, m)?)?;
    m.add_function(wrap_pyfunction!(get_batched_frame_gil, m)?)?;
    m.add_function(wrap_pyfunction!(get_batch_gil, m)?)?;
    m.add_function(wrap_pyfunction!(apply_updates_gil, m)?)?;
    m.add_function(wrap_pyfunction!(move_as_is_gil, m)?)?;
    m.add_function(wrap_pyfunction!(move_and_pack_frames_gil, m)?)?;
    m.add_function(wrap_pyfunction!(move_and_unpack_batch_gil, m)?)?;

    Ok(())
}
