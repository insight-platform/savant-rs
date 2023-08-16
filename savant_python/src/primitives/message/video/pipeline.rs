use crate::primitives::message::video::frame_update::VideoFrameUpdate;
use crate::primitives::message::video::match_query::MatchQuery;
use crate::primitives::message::video::objects_view::VideoObjectsView;
use crate::primitives::{VideoFrame, VideoFrameBatch};
use crate::release_gil;
use crate::utils::otlp::TelemetrySpan;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use savant_core::rust;
use std::collections::HashMap;

#[pyclass]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum VideoPipelineStagePayloadType {
    Frame,
    Batch,
}

impl From<VideoPipelineStagePayloadType> for rust::PipelineStagePayloadType {
    fn from(p: VideoPipelineStagePayloadType) -> Self {
        match p {
            VideoPipelineStagePayloadType::Frame => rust::PipelineStagePayloadType::Frame,
            VideoPipelineStagePayloadType::Batch => rust::PipelineStagePayloadType::Batch,
        }
    }
}

impl From<rust::PipelineStagePayloadType> for VideoPipelineStagePayloadType {
    fn from(p: rust::PipelineStagePayloadType) -> Self {
        match p {
            rust::PipelineStagePayloadType::Frame => VideoPipelineStagePayloadType::Frame,
            rust::PipelineStagePayloadType::Batch => VideoPipelineStagePayloadType::Batch,
        }
    }
}

#[pymodule]
pub(crate) fn pipeline(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<VideoPipelineStagePayloadType>()?;
    m.add_class::<VideoPipeline>()?;
    Ok(())
}

#[pyclass]
#[derive(Debug)]
struct VideoPipeline(rust::Pipeline);

#[pymethods]
impl VideoPipeline {
    #[new]
    fn new(name: String) -> Self {
        let p = rust::Pipeline::default();
        p.set_root_span_name(name);
        Self(p)
    }

    #[getter]
    fn memory_handle(&self) -> usize {
        self.0.memory_handle()
    }

    /// Sets the root span name.
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///   The name of the root span.
    ///
    #[setter]
    fn set_root_span_name(&self, name: String) {
        self.0.set_root_span_name(name);
    }

    /// Returns the root span name.
    ///
    #[getter]
    fn get_root_span_name(&self) -> String {
        self.0.get_root_span_name()
    }

    /// Set sampling
    ///
    #[setter]
    fn set_sampling_period(&self, period: i64) {
        self.0.set_sampling_period(period);
    }

    /// Get sampling
    ///
    #[getter]
    fn get_sampling_period(&self) -> i64 {
        self.0.get_sampling_period()
    }

    /// Adds a stage to the pipeline.
    ///
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
    fn add_stage(&self, name: &str, stage_type: VideoPipelineStagePayloadType) -> PyResult<()> {
        self.0
            .add_stage(name, stage_type.into())
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
    fn get_stage_type(&self, name: &str) -> Option<VideoPipelineStagePayloadType> {
        self.0.get_stage_type(name).map(|t| t.clone().into())
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
        stage: &str,
        frame_id: i64,
        update: VideoFrameUpdate,
    ) -> PyResult<()> {
        self.0
            .add_frame_update(stage, frame_id, update.0)
            .map_err(|e| PyValueError::new_err(e.to_string()))
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
        stage: &str,
        batch_id: i64,
        frame_id: i64,
        update: VideoFrameUpdate,
    ) -> PyResult<()> {
        self.0
            .add_batched_frame_update(stage, batch_id, frame_id, update.0)
            .map_err(|e| PyValueError::new_err(e.to_string()))
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
    fn add_frame(&self, stage_name: &str, frame: VideoFrame) -> PyResult<i64> {
        self.0
            .add_frame(stage_name, frame.0)
            .map_err(|e| PyValueError::new_err(e.to_string()))
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
    /// parent_ctx : :py:class:`savant_rs.utils.TelemetrySpan`
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
    pub fn add_frame_with_telemetry(
        &self,
        stage_name: &str,
        frame: VideoFrame,
        parent_span: &TelemetrySpan,
    ) -> PyResult<i64> {
        self.0
            .add_frame_with_telemetry(stage_name, frame.0, parent_span.0.clone())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Deletes a frame or a batch from the stage.
    ///
    /// GIL management: the function is GIL-free.
    ///
    /// Parameters
    /// ----------
    /// id : int
    ///   The id of the frame or batch to delete.
    ///
    /// Returns
    /// -------
    /// dict[int, :py:class:`savant_rs.utils.TelemetrySpan`]
    ///   The ids of the frames and their telemetry contexts.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///   If the stage does not exist. If the frame or batch does not exist.
    ///
    fn delete(&self, id: i64) -> PyResult<HashMap<i64, TelemetrySpan>> {
        let res = self.0.delete(id);
        match res {
            Ok(h) => Ok(h
                .into_iter()
                .map(|(k, v)| (k, TelemetrySpan::from_context(v)))
                .collect()),
            Err(e) => Err(PyValueError::new_err(e.to_string())),
        }
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
    fn get_stage_queue_len(&self, stage_name: &str) -> PyResult<usize> {
        self.0
            .get_stage_queue_len(stage_name)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    /// Retrieves an independent frame from a specified stage.
    ///
    /// GIL management: the function is GIL-free.
    ///
    /// Parameters
    /// ----------
    /// frame_id : int
    ///   The id of the frame.
    ///
    /// Returns
    /// -------
    /// (:py:class:`savant_rs.primitives.VideoFrameProxy`, :py:class:`savant_rs.utils.TelemetrySpan`)
    ///   The frame.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///   If the stage does not exist or is not of type independent frames. If the frame does not exist.
    ///
    fn get_independent_frame(&self, frame_id: i64) -> PyResult<(VideoFrame, TelemetrySpan)> {
        self.0
            .get_independent_frame(frame_id)
            .map(|(f, c)| (VideoFrame(f), TelemetrySpan::from_context(c)))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    /// Retrieves a batched frame from a specified stage.
    ///
    /// GIL management: the function is GIL-free.
    ///
    /// Parameters
    /// ----------
    /// batch_id : int
    ///   The id of the batch.
    /// frame_id : int
    ///   The id of the frame.
    ///
    /// Returns
    /// -------
    /// (:py:class:`savant_rs.primitives.VideoFrameProxy`, :py:class:`savant_rs.utils.TelemetrySpan`)
    ///   The frame and the OTLP propagation context corresponding to the current phase of processing.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///   If the stage does not exist or is not of type batches. If the batch or the frame do not exist.
    ///
    fn get_batched_frame(
        &self,
        batch_id: i64,
        frame_id: i64,
    ) -> PyResult<(VideoFrame, TelemetrySpan)> {
        self.0
            .get_batched_frame(batch_id, frame_id)
            .map(|(f, c)| (VideoFrame(f), TelemetrySpan::from_context(c)))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    /// Retrieves a batch from a specified stage.
    ///
    /// GIL management: the function is GIL-free.
    ///
    /// Parameters
    /// ----------
    /// batch_id : int
    ///   The id of the batch.
    ///
    /// Returns
    /// -------
    /// (:py:class:`savant_rs.primitives.VideoFrameBatch`, Dict[int, :py:class:`savant_rs.utils.TelemetrySpan`])
    ///   The batch and propagation contexts for every frame.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///   If the stage does not exist or is not of type batches. If the batch does not exist.
    ///
    fn get_batch(&self, batch_id: i64) -> PyResult<(VideoFrameBatch, HashMap<i64, TelemetrySpan>)> {
        self.0
            .get_batch(batch_id)
            .map(|(b, c)| {
                (
                    VideoFrameBatch(b),
                    c.into_iter()
                        .map(|(k, v)| (k, TelemetrySpan::from_context(v)))
                        .collect(),
                )
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    /// Applies the updates to the frames and batches of a stage.
    ///
    /// GIL management: the function is GIL-free.
    ///
    /// Parameters
    /// ----------
    /// id : int
    ///   The id of the frame or batch to apply updates for.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///   If the stage does not exist. If the frame or batch does not exist.
    ///
    #[pyo3(name = "apply_updates")]
    #[pyo3(signature = (id, no_gil = true))]
    fn apply_updates_gil(&self, id: i64, no_gil: bool) -> PyResult<()> {
        release_gil!(no_gil, || {
            self.0
                .apply_updates(id)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })
    }

    /// Clears the updates to the frames and batches of a stage.
    ///
    /// GIL management: the function is GIL-free.
    ///
    /// Parameters
    /// ----------
    /// id : int
    ///   The id of the frame or batch to clear updates for.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///   If the stage does not exist. If the frame or batch does not exist.
    ///
    fn clear_updates(&self, id: i64) -> PyResult<()> {
        self.0
            .clear_updates(id)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Moves frames or batches from a stage to another. The dest stage must be the same time as the source stage.
    ///
    /// GIL management: the function is GIL-free.
    ///
    /// Parameters
    /// ----------
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

    #[pyo3(name = "move_as_is")]
    #[pyo3(signature = (dest_stage_name, object_ids, no_gil = true))]
    fn move_as_is_gil(
        &self,
        dest_stage_name: &str,
        object_ids: Vec<i64>,
        no_gil: bool,
    ) -> PyResult<()> {
        release_gil!(no_gil, || {
            self.0
                .move_as_is(dest_stage_name, object_ids)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })
    }
    /// Moves frames from the stage with independent frames to the stage with batches.
    ///
    /// GIL management: the function is GIL-free.
    ///
    /// Parameters
    /// ----------
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
    #[pyo3(name = "move_and_pack_frames")]
    #[pyo3(signature = (dest_stage_name, frame_ids, no_gil = true))]
    fn move_and_pack_frames_gil(
        &self,
        dest_stage_name: &str,
        frame_ids: Vec<i64>,
        no_gil: bool,
    ) -> PyResult<i64> {
        release_gil!(no_gil, || {
            self.0
                .move_and_pack_frames(dest_stage_name, frame_ids)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })
    }
    /// Moves a batch from the stage with batches to the stage with independent frames.
    ///
    /// GIL management: the function is GIL-free.
    ///
    /// Parameters
    /// ----------
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
    #[pyo3(name = "move_and_unpack_batch")]
    #[pyo3(signature = (dest_stage_name, batch_id, no_gil = true))]
    fn move_and_unpack_batch_gil(
        &self,
        dest_stage_name: &str,
        batch_id: i64,
        no_gil: bool,
    ) -> PyResult<HashMap<String, i64>> {
        release_gil!(no_gil, || {
            self.0
                .move_and_unpack_batch(dest_stage_name, batch_id)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })
    }

    #[pyo3(name = "access_objects")]
    #[pyo3(signature = (frame_id, query, no_gil = true))]
    pub fn access_objects_gil(
        &self,
        frame_id: i64,
        query: &MatchQuery,
        no_gil: bool,
    ) -> PyResult<HashMap<i64, VideoObjectsView>> {
        release_gil!(no_gil, || {
            self.0
                .access_objects(frame_id, &query.0)
                .map(|result| {
                    result
                        .into_iter()
                        .map(|(k, v)| (k, VideoObjectsView::from(v)))
                        .collect()
                })
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })
    }
}
