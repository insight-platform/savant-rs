use super::VideoPipeline;
use crate::primitives::message::video::pipeline::VideoPipelineStagePayloadType;
use crate::primitives::{VideoFrameBatch, VideoFrameProxy, VideoFrameUpdate};
use crate::utils::propagation_context::PropagationContext;
use crate::utils::python::release_gil;
use lazy_static::lazy_static;
use parking_lot::{const_mutex, Mutex};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;

lazy_static! {
    static ref PIPELINE: Mutex<VideoPipeline> = const_mutex(VideoPipeline::default());
}

/// Adds a stage to the pipeline.
///
/// GIL management: the function is GIL-free.
///
/// Parameters
/// ----------
/// name : str
///   The name of the stage.
/// stage_type : :py:class:`VideoPipelineStagePayloadType`
///   The type of the stage. Either independent frames or batches.
///
/// Raises
/// ------
/// ValueError
///   If the stage name is already in use.
///
#[pyfunction]
#[pyo3(name = "add_stage")]
fn add_stage_gil(name: String, stage_type: VideoPipelineStagePayloadType) -> PyResult<()> {
    release_gil(|| PIPELINE.lock().add_stage(&name, stage_type))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Retrieves the type of a stage.
///
/// GIL management: the function is GIL-free.
///
/// Parameters
/// ----------
/// name : str
///  The name of the stage.
///
/// Returns
/// -------
/// :py:class:`VideoPipelineStagePayloadType`
///   The type of the stage. Either independent frames or batches.
/// None
///  If the stage does not exist.
///
#[pyfunction]
#[pyo3(name = "get_stage_type")]
fn get_stage_type_gil(name: String) -> Option<VideoPipelineStagePayloadType> {
    release_gil(|| PIPELINE.lock().get_stage_type(&name).cloned())
}

/// Adds a frame update to the independent frame.
///
/// GIL management: the function is GIL-free.
///
/// Parameters
/// ----------
/// stage : str
///   The name of the stage. Must be a stage of type independent frames.
/// frame_id : int
///   The id of the frame.
/// update : :py:class:`savant_rs.primitives.VideoFrameUpdate`
///   The update to enqueue for the frame. The updates are not applied immediately, but when requested with :py:func:`apply_updates`.
///
/// Raises
/// ------
/// ValueError
///  If the stage does not exist or is not of type independent frames. If the frame does not exist.
///
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

/// Adds a frame update to the batched frame.
///
/// GIL management: the function is GIL-free.
///
/// Parameters
/// ----------
/// stage : str
///   The name of the stage. Must be a stage of type batches.
/// batch_id : int
///   The id of the batch.
/// frame_id : int
///   The id of the frame.
/// update : :py:class:`savant_rs.primitives.VideoFrameUpdate`
///   The update to enqueue for the frame. The updates are not applied immediately, but when requested with :py:func:`apply_updates`.
///
/// Raises
/// ------
/// ValueError
///   If the stage does not exist or is not of type batches. If the batch or the frame do not exist.
///
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

/// Adds a frame to the stage.
///
/// GIL management: the function is GIL-free.
///
/// Parameters
/// ----------
/// stage_name : str
///   The name of the stage. Must be a stage of type independent frames.
/// frame : :py:class:`savant_rs.primitives.VideoFrameProxy`
///   The frame to add.
///
/// Returns
/// -------
/// int
///   The id of the frame.
///
/// Raises
/// ------
/// ValueError
///   If the stage does not exist or is not of type independent frames.
///
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

/// Deletes a frame or a batch from the stage.
///
/// GIL management: the function is GIL-free.
///
/// Parameters
/// ----------
/// stage_name : str
///   The name of the stage.
/// id : int
///   The id of the frame or batch to delete.
///
/// Raises
/// ------
/// ValueError
///   If the stage does not exist. If the frame or batch does not exist.
///
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

/// Retrieves the length of the queue of a stage.
///
/// GIL management: the function is GIL-free.
///
/// Parameters
/// ----------
/// stage_name : str
///   The name of the stage.
///
/// Returns
/// -------
/// int
///   The length of the queue of the stage.
///
/// Raises
/// ------
/// ValueError
///   If the stage does not exist.
///
#[pyfunction]
#[pyo3(name = "get_stage_queue_len")]
fn get_stage_queue_len_gil(stage_name: String) -> PyResult<usize> {
    release_gil(|| {
        PIPELINE
            .lock()
            .get_stage_queue_len(&stage_name)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })
}

/// Retrieves an independent frame from a specified stage.
///
/// GIL management: the function is GIL-free.
///
/// Parameters
/// ----------
/// stage : str
///   The name of the stage.
/// frame_id : int
///   The id of the frame.
///
/// Returns
/// -------
/// :py:class:`savant_rs.primitives.VideoFrameProxy`
///   The frame.
///
/// Raises
/// ------
/// ValueError
///   If the stage does not exist or is not of type independent frames. If the frame does not exist.
///
#[pyfunction]
#[pyo3(name = "get_independent_frame")]
fn get_independent_frame_gil(
    stage: String,
    frame_id: i64,
) -> PyResult<(VideoFrameProxy, PropagationContext)> {
    release_gil(|| {
        PIPELINE
            .lock()
            .get_independent_frame(&stage, frame_id)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })
}

/// Retrieves a batched frame from a specified stage.
///
/// GIL management: the function is GIL-free.
///
/// Parameters
/// ----------
/// stage : str
///   The name of the stage.
/// batch_id : int
///   The id of the batch.
/// frame_id : int
///   The id of the frame.
///
/// Returns
/// -------
/// :py:class:`savant_rs.primitives.VideoFrameProxy`
///   The frame.
///
/// Raises
/// ------
/// ValueError
///   If the stage does not exist or is not of type batches. If the batch or the frame do not exist.
///
#[pyfunction]
#[pyo3(name = "get_batched_frame")]
fn get_batched_frame_gil(
    stage: String,
    batch_id: i64,
    frame_id: i64,
) -> PyResult<(VideoFrameProxy, PropagationContext)> {
    release_gil(|| {
        PIPELINE
            .lock()
            .get_batched_frame(&stage, batch_id, frame_id)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })
}

/// Retrieves a batch from a specified stage.
///
/// GIL management: the function is GIL-free.
///
/// Parameters
/// ----------
/// stage : str
///   The name of the stage.
/// batch_id : int
///   The id of the batch.
///
/// Returns
/// -------
/// :py:class:`savant_rs.primitives.VideoFrameBatch`
///   The batch.
///
/// Raises
/// ------
/// ValueError
///   If the stage does not exist or is not of type batches. If the batch does not exist.
///
#[pyfunction]
#[pyo3(name = "get_batch")]
fn get_batch_gil(
    stage: String,
    batch_id: i64,
) -> PyResult<(VideoFrameBatch, HashMap<i64, PropagationContext>)> {
    release_gil(|| {
        PIPELINE
            .lock()
            .get_batch(&stage, batch_id)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })
}

/// Applies the updates to the frames and batches of a stage.
///
/// GIL management: the function is GIL-free.
///
/// Parameters
/// ----------
/// stage : str
///   The name of the stage.
/// id : int
///   The id of the frame or batch to apply updates for.
///
/// Raises
/// ------
/// ValueError
///   If the stage does not exist. If the frame or batch does not exist.
///
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

/// Moves frames or batches from a stage to another. The dest stage must be the same time as the source stage.
///
/// GIL management: the function is GIL-free.
///
/// Parameters
/// ----------
/// source_stage_name : str
///   The name of the source stage.
/// dest_stage_name : str
///   The name of the destination stage.
/// object_ids : List[int]
///   The ids of the frames or batches to move.
///
/// Raises
/// ------
/// ValueError
///   If the source stage does not exist. If the destination stage does not exist.
///   If the source stage and the destination stage are not of the same type.
///   If the frame or batch does not exist.
///
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

/// Moves frames from the stage with independent frames to the stage with batches.
///
/// GIL management: the function is GIL-free.
///
/// Parameters
/// ----------
/// source_stage_name : str
///   The name of the source stage.
/// dest_stage_name : str
///   The name of the destination stage.
/// frame_ids : List[int]
///   The ids of the frames to move.
///
/// Returns
/// -------
/// int
///   The id of the batch.
///
/// Raises
/// ------
/// ValueError
///   If the source stage does not exist or destination stage does not exist.
///   If the source stage is not of type independent frames or the destination stage is not of type batches.
///   If the frame does not exist.
///
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

/// Moves a batch from the stage with batches to the stage with independent frames.
///
/// GIL management: the function is GIL-free.
///
/// Parameters
/// ----------
/// source_stage_name : str
///   The name of the source stage.
/// dest_stage_name : str
///   The name of the destination stage.
/// batch_id : int
///   The id of the batch to move.
///
/// Returns
/// -------
/// Dict[str, int]
///   The map of stream_id: id for the frames unpacked from the batch.
///
/// Raises
/// ------
/// ValueError
///   If the source stage does not exist or destination stage does not exist.
///   If the source stage is not of type batches or the destination stage is not of type independent frames.
///   If the batch does not exist.
///
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

    m.add_function(wrap_pyfunction!(add_stage_gil, m)?)?;
    m.add_function(wrap_pyfunction!(get_stage_type_gil, m)?)?;
    m.add_function(wrap_pyfunction!(add_frame_update_gil, m)?)?;
    m.add_function(wrap_pyfunction!(add_batched_frame_update_gil, m)?)?;
    m.add_function(wrap_pyfunction!(add_frame_gil, m)?)?;
    m.add_function(wrap_pyfunction!(delete_gil, m)?)?;
    m.add_function(wrap_pyfunction!(get_stage_queue_len_gil, m)?)?;
    m.add_function(wrap_pyfunction!(get_independent_frame_gil, m)?)?;
    m.add_function(wrap_pyfunction!(get_batched_frame_gil, m)?)?;
    m.add_function(wrap_pyfunction!(get_batch_gil, m)?)?;
    m.add_function(wrap_pyfunction!(apply_updates_gil, m)?)?;
    m.add_function(wrap_pyfunction!(move_as_is_gil, m)?)?;
    m.add_function(wrap_pyfunction!(move_and_pack_frames_gil, m)?)?;
    m.add_function(wrap_pyfunction!(move_and_unpack_batch_gil, m)?)?;

    Ok(())
}
