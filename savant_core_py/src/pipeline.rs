use crate::match_query::MatchQuery;
use crate::primitives::attribute_value::AttributeValue;
use crate::primitives::batch::VideoFrameBatch;
use crate::primitives::frame::VideoFrame;
use crate::primitives::frame_update::VideoFrameUpdate;
use crate::primitives::objects_view::VideoObjectsView;
use crate::release_gil;
use crate::utils::otlp::TelemetrySpan;
use pyo3::exceptions::{PySystemError, PyValueError};
use pyo3::prelude::*;
use savant_core::pipeline::stage_function_loader::load_stage_function_plugin as rust_load_stage_function_plugin;
use savant_core::pipeline::PipelineStageFunction as RustPipelineStageFunction;
use savant_core::pipeline::PluginParams;
use savant_core::rust;
use std::cell::Cell;
use std::collections::HashMap;

#[pyclass]
pub struct StageFunction(Cell<Option<Box<dyn RustPipelineStageFunction>>>);

impl StageFunction {
    pub fn new(f: Box<dyn RustPipelineStageFunction>) -> Self {
        Self(Cell::new(Some(f)))
    }
}

impl Clone for StageFunction {
    fn clone(&self) -> Self {
        let f = self.0.take();
        Self(Cell::new(f))
    }
}

#[pyfunction]
pub fn handle_psf(f: StageFunction) {
    let _ = f;
}

#[pymethods]
impl StageFunction {
    #[staticmethod]
    fn none() -> Self {
        Self(Cell::new(None))
    }
}

#[pyfunction]
pub fn load_stage_function_plugin(
    libname: &str,
    init_name: &str,
    plugin_name: &str,
    params: HashMap<String, AttributeValue>,
) -> PyResult<StageFunction> {
    let params = params
        .into_iter()
        .map(|(k, v)| (k, v.0))
        .collect::<hashbrown::HashMap<_, _>>();
    let params = PluginParams { params };

    rust_load_stage_function_plugin(libname, init_name, plugin_name, params)
        .map(|f| StageFunction(Cell::new(Some(f))))
        .map_err(|e| PySystemError::new_err(e.to_string()))
}

/// Defines which type of payload a stage handles.
///
#[pyclass]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum VideoPipelineStagePayloadType {
    Frame,
    Batch,
}

#[pyclass]
pub enum FrameProcessingStatRecordType {
    Initial,
    Frame,
    Timestamp,
}

impl From<FrameProcessingStatRecordType> for rust::FrameProcessingStatRecordType {
    fn from(t: FrameProcessingStatRecordType) -> Self {
        match t {
            FrameProcessingStatRecordType::Initial => rust::FrameProcessingStatRecordType::Initial,
            FrameProcessingStatRecordType::Frame => rust::FrameProcessingStatRecordType::Frame,
            FrameProcessingStatRecordType::Timestamp => {
                rust::FrameProcessingStatRecordType::Timestamp
            }
        }
    }
}

impl From<rust::FrameProcessingStatRecordType> for FrameProcessingStatRecordType {
    fn from(t: rust::FrameProcessingStatRecordType) -> Self {
        match t {
            rust::FrameProcessingStatRecordType::Initial => FrameProcessingStatRecordType::Initial,
            rust::FrameProcessingStatRecordType::Frame => FrameProcessingStatRecordType::Frame,
            rust::FrameProcessingStatRecordType::Timestamp => {
                FrameProcessingStatRecordType::Timestamp
            }
        }
    }
}

#[pyclass]
pub struct StageStat(rust::StageStat);

#[pymethods]
impl StageStat {
    #[getter]
    fn stage_name(&self) -> String {
        self.0.stage_name.clone()
    }

    #[getter]
    fn queue_length(&self) -> usize {
        self.0.queue_length
    }

    #[getter]
    fn frame_counter(&self) -> usize {
        self.0.frame_counter
    }

    #[getter]
    fn object_counter(&self) -> usize {
        self.0.object_counter
    }

    #[getter]
    fn batch_counter(&self) -> usize {
        self.0.batch_counter
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }

    fn __str__(&self) -> String {
        format!("{:#?}", self.0)
    }
}

#[pyclass]
pub struct FrameProcessingStatRecord(rust::FrameProcessingStatRecord);

#[pymethods]
impl FrameProcessingStatRecord {
    #[getter]
    fn id(&self) -> i64 {
        self.0.id
    }

    #[getter]
    fn ts(&self) -> i64 {
        self.0.ts
    }

    #[getter]
    fn frame_no(&self) -> usize {
        self.0.frame_no
    }

    #[getter]
    fn record_type(&self) -> FrameProcessingStatRecordType {
        self.0.record_type.clone().into()
    }

    #[getter]
    fn object_counter(&self) -> usize {
        self.0.object_counter
    }

    #[getter]
    fn stage_stats(&self) -> Vec<StageStat> {
        self.0
            .stage_stats
            .clone()
            .into_iter()
            .map(StageStat)
            .collect()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }

    fn __str__(&self) -> String {
        format!("{:#?}", self.0)
    }
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

/// A video pipeline.
///
#[pyclass]
#[pyo3(name = "VideoPipeline")]
#[derive(Debug)]
pub struct Pipeline(rust::Pipeline);

#[pyclass]
#[pyo3(name = "VideoPipelineConfiguration")]
#[derive(Debug, Clone)]
pub struct PipelineConfiguration(rust::PipelineConfiguration);

#[pymethods]
impl PipelineConfiguration {
    #[new]
    pub fn new() -> PyResult<Self> {
        Ok(Self(
            rust::PipelineConfigurationBuilder::default()
                .build()
                .map_err(|e| {
                    PyValueError::new_err(format!("Failed to create pipeline configuration: {}", e))
                })?,
        ))
    }

    #[setter]
    pub fn append_frame_meta_to_otlp_span(&mut self, v: bool) {
        self.0.append_frame_meta_to_otlp_span = v;
    }

    #[setter]
    pub fn timestamp_period(&mut self, v: Option<i64>) {
        self.0.timestamp_period = v;
    }

    #[setter]
    pub fn frame_period(&mut self, v: Option<i64>) {
        self.0.frame_period = v;
    }

    #[setter]
    pub fn collection_history(&mut self, v: usize) {
        self.0.collection_history = v;
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }

    fn __str__(&self) -> String {
        format!("{:#?}", self.0)
    }
}

#[pymethods]
impl Pipeline {
    #[new]
    fn new(
        name: String,
        stages: Vec<(
            String,
            VideoPipelineStagePayloadType,
            StageFunction,
            StageFunction,
        )>,
        configuration: PipelineConfiguration,
    ) -> PyResult<Self> {
        let stages = stages
            .into_iter()
            .map(|(n, t, i, e)| {
                let ingress = i.0.take();
                let egress = e.0.take();
                (n, t.into(), ingress, egress)
            })
            .collect();
        let p = rust::Pipeline::new(stages, configuration.0)
            .map_err(|e| PyValueError::new_err(format!("Failed to create pipeline: {}", e)))?;
        p.set_root_span_name(name)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self(p))
    }

    pub fn get_stat_records(&self, max_n: usize) -> Vec<FrameProcessingStatRecord> {
        self.0
            .get_stat_records(max_n)
            .into_iter()
            .map(FrameProcessingStatRecord)
            .collect()
    }

    pub fn get_stat_records_newer_than(&self, id: i64) -> Vec<FrameProcessingStatRecord> {
        self.0
            .get_stat_records_newer_than(id)
            .into_iter()
            .map(FrameProcessingStatRecord)
            .collect()
    }

    pub fn log_final_fps(&self) {
        self.0.log_final_fps();
    }

    /// Clears the ordering for source, called on dead stream eviction.
    ///
    /// Parameters
    /// ----------
    /// source_id : str
    ///   The id of the source.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///   If the source does not exist.
    ///
    pub fn clear_source_ordering(&self, source_id: &str) -> PyResult<()> {
        self.0
            .clear_source_ordering(source_id)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Allows receiving a raw pointer to Rust inner Pipeline struct.
    ///
    #[getter]
    fn memory_handle(&self) -> usize {
        self.0.memory_handle()
    }

    /// Returns the root span name.
    ///
    #[getter]
    fn get_root_span_name(&self) -> String {
        self.0.get_root_span_name()
    }

    /// Set sampling. Sampling is used to reduce the number of spans sent to an OTLP collector.
    /// By default, it is set to 0, which means that spans are only produced for propagated telemetry.
    ///
    /// Params
    /// ------
    /// period : int
    ///   The sampling period. If set to 0, no sampling is performed.
    ///   Set it to a high number (e.g. 100, 1000) for production, 1 for development purposes to trace every frame.
    ///
    #[setter]
    fn set_sampling_period(&self, period: i64) -> PyResult<()> {
        self.0.set_sampling_period(period).map_err(|e| {
            PyValueError::new_err(format!(
                "Failed to set sampling period to {}: {}",
                period, e
            ))
        })
    }

    /// Get sampling configured for the pipeline.
    ///
    /// Returns
    /// -------
    /// int
    ///   The sampling period.
    ///
    #[getter]
    fn get_sampling_period(&self) -> i64 {
        self.0.get_sampling_period()
    }

    /// Retrieves the type of a stage.
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
    ///
    /// Raises
    /// ------
    /// ValueError
    ///   If the stage does not exist.
    ///
    fn get_stage_type(&self, name: &str) -> PyResult<VideoPipelineStagePayloadType> {
        self.0
            .get_stage_type(name)
            .map(VideoPipelineStagePayloadType::from)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    /// Adds a frame update to the independent frame.
    ///
    /// Parameters
    /// ----------
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
    fn add_frame_update(&self, frame_id: i64, update: VideoFrameUpdate) -> PyResult<()> {
        self.0
            .add_frame_update(frame_id, update.0)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    /// Adds a frame update to the batched frame.
    ///
    /// GIL management: the function is GIL-free.
    ///
    /// Parameters
    /// ----------
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
        batch_id: i64,
        frame_id: i64,
        update: VideoFrameUpdate,
    ) -> PyResult<()> {
        self.0
            .add_batched_frame_update(batch_id, frame_id, update.0)
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
    ) -> PyResult<Vec<i64>> {
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
