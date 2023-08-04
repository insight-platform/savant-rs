use crate::primitives::message::video::pipeline::VideoPipeline as VideoPipelineRs;
use crate::primitives::message::video::pipeline::VideoPipelineStagePayloadType;
use crate::primitives::{VideoFrameBatch, VideoFrameProxy, VideoFrameUpdate};
use crate::utils::otlp::{OTLPSpan, PropagatedContext};
use crate::utils::python::release_gil;
use parking_lot::Mutex;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

#[pymodule]
pub(crate) fn pipeline(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<VideoPipelineStagePayloadType>()?;
    m.add_class::<VideoPipeline>()?;
    Ok(())
}

#[pyclass]
#[derive(Debug, Clone)]
struct VideoPipeline(Arc<Mutex<VideoPipelineRs>>);

#[pymethods]
impl VideoPipeline {
    #[new]
    fn new(name: String) -> Self {
        let p = VideoPipeline(Arc::new(Mutex::new(VideoPipelineRs::default())));
        p.set_root_span_name(name);
        p
    }

    /// Sets the root span name.
    ///
    /// GIL management: the function is GIL-free.
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///   The name of the root span.
    ///
    fn set_root_span_name(&self, name: String) {
        release_gil(|| self.0.lock().set_root_span_name(name));
    }

    /// Returns the root span name.
    ///
    fn get_root_span_name(&self) -> String {
        self.0.lock().get_root_span_name()
    }

    /// Adds a stage to the pipeline.
    ///
    /// GIL management: the function is GIL-free.
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///  The name of the stage.
    /// stage_type : :py:class:`VideoPipelineStagePayloadType`
    ///  The type of the stage. Either independent frames or batches.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///   If the stage name is already in use.
    ///
    fn add_stage(&self, name: String, stage_type: VideoPipelineStagePayloadType) -> PyResult<()> {
        release_gil(|| self.0.lock().add_stage(&name, stage_type))
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
    fn get_stage_type(&self, name: String) -> Option<VideoPipelineStagePayloadType> {
        self.0.lock().get_stage_type(&name).cloned()
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
    fn add_frame_update(
        &self,
        stage: String,
        frame_id: i64,
        update: VideoFrameUpdate,
    ) -> PyResult<()> {
        release_gil(|| {
            self.0
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
    fn add_batched_frame_update(
        &self,
        stage: String,
        batch_id: i64,
        frame_id: i64,
        update: VideoFrameUpdate,
    ) -> PyResult<()> {
        release_gil(|| {
            self.0
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
    fn add_frame(&self, stage_name: String, frame: VideoFrameProxy) -> PyResult<i64> {
        release_gil(|| {
            self.0
                .lock()
                .add_frame(&stage_name, frame)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })
    }

    /// Adds a frame to the stage with an OTLP parent context.
    ///
    /// GIL management: the function is GIL-free.
    ///
    /// Parameters
    /// ----------
    /// stage_name : str
    ///   The name of the stage. Must be a stage of type independent frames.
    /// frame : :py:class:`savant_rs.primitives.VideoFrameProxy`
    ///   The frame to add.
    /// parent_ctx : :py:class:`savant_rs.primitives.PropagatedContext`
    ///   The parent context to add to the frame.
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
    pub fn add_frame_with_remote_telemetry(
        &self,
        stage_name: &str,
        frame: VideoFrameProxy,
        remote_telemetry_ctx: PropagatedContext,
    ) -> PyResult<i64> {
        release_gil(|| {
            self.0
                .lock()
                .add_frame_with_remote_telemetry(stage_name, frame, remote_telemetry_ctx)
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
    fn delete(&self, stage_name: String, id: i64) -> PyResult<HashMap<i64, OTLPSpan>> {
        release_gil(|| {
            let res = self.0.lock().delete(&stage_name, id);
            match res {
                Ok(h) => Ok(h.into_iter().map(|(k, v)| (k, OTLPSpan(v))).collect()),
                Err(e) => Err(PyValueError::new_err(e.to_string())),
            }
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
    fn get_stage_queue_len(&self, stage_name: String) -> PyResult<usize> {
        release_gil(|| {
            self.0
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
    /// (:py:class:`savant_rs.primitives.VideoFrameProxy`, :py:class:`savant_rs.utils.PropagationContext`)
    ///   The frame.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///   If the stage does not exist or is not of type independent frames. If the frame does not exist.
    ///
    fn get_independent_frame(
        &self,
        stage: String,
        frame_id: i64,
    ) -> PyResult<(VideoFrameProxy, PropagatedContext)> {
        release_gil(|| {
            self.0
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
    /// (:py:class:`savant_rs.primitives.VideoFrameProxy`, :py:class:`savant_rs.utils.PropagationContext`)
    ///   The frame and the OTLP propagation context corresponding to the current phase of processing.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///   If the stage does not exist or is not of type batches. If the batch or the frame do not exist.
    ///
    fn get_batched_frame(
        &self,
        stage: String,
        batch_id: i64,
        frame_id: i64,
    ) -> PyResult<(VideoFrameProxy, PropagatedContext)> {
        release_gil(|| {
            self.0
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
    /// (:py:class:`savant_rs.primitives.VideoFrameBatch`, Dict[int, :py:class:`savant_rs.utils.PropagationContext])
    ///   The batch and propagation contexts for every frame.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///   If the stage does not exist or is not of type batches. If the batch does not exist.
    ///
    fn get_batch(
        &self,
        stage: String,
        batch_id: i64,
    ) -> PyResult<(VideoFrameBatch, HashMap<i64, PropagatedContext>)> {
        release_gil(|| {
            self.0
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
    fn apply_updates(&self, stage: String, id: i64) -> PyResult<()> {
        release_gil(|| {
            self.0
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
    fn move_as_is(
        &self,
        source_stage_name: String,
        dest_stage_name: String,
        object_ids: Vec<i64>,
    ) -> PyResult<()> {
        release_gil(|| {
            self.0
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
    fn move_and_pack_frames(
        &self,
        source_stage_name: String,
        dest_stage_name: String,
        frame_ids: Vec<i64>,
    ) -> PyResult<i64> {
        release_gil(|| {
            self.0
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
    fn move_and_unpack_batch(
        &self,
        source_stage_name: String,
        dest_stage_name: String,
        batch_id: i64,
    ) -> PyResult<HashMap<String, i64>> {
        release_gil(|| {
            self.0
                .lock()
                .move_and_unpack_batch(&source_stage_name, &dest_stage_name, batch_id)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })
    }
}
