//! PyO3 bindings for the [`NvInferBatchingOperator`](nvinfer::NvInferBatchingOperator) batching layer.

use super::config::PyNvInferConfig;
use super::enums::PyDataType;
use super::output::PyInferDims;
use super::pipeline::take_shared_buffer;
use super::roi::PyRoiKind;
use crate::deepstream::enums::{from_rust_id_kind, to_rust_id_kind, PySavantIdMetaKind};
use crate::deepstream::PySharedBuffer;
use crate::primitives::bbox::RBBox as PyRBBox;
use crate::primitives::frame::VideoFrame;
use deepstream_buffers::SavantIdMetaKind;
use numpy::{PyArray2, PyReadonlyArray2};
use nvinfer::{
    BatchFormationCallback, BatchFormationResult, NvInferBatchingOperator,
    NvInferBatchingOperatorConfig, OperatorInferenceOutput, OperatorOutput, RoiKind,
    SealedDeliveries,
};
use parking_lot::Mutex;
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::sync::Arc;
use std::time::Duration;

/// Shared lifetime anchor for tensor pointers borrowed from a
/// [`OperatorInferenceOutput`].  Mirrors the `SharedOutput` pattern in
/// [`super::output`] but anchored to the operator output type.
type SharedOperatorOutput = Arc<Mutex<Option<OperatorInferenceOutput>>>;

// ─── NvInferBatchingOperatorConfig ──────────────────────────────────────────────

/// Configuration for the NvInferBatchingOperator batching layer.
///
/// Embeds a full ``NvInferConfig`` which is forwarded to the inner NvInfer
/// pipeline.  The GPU device ID for batch construction is taken from
/// ``nvinfer_config.gpu_id``.
///
/// Args:
///     max_batch_size (int): Maximum frames per batch.
///     max_batch_wait_ms (int): Maximum time (milliseconds) to wait before
///         submitting a partial batch.
///     nvinfer_config (NvInferConfig): Configuration forwarded to the inner
///         NvInfer engine.
///     pending_batch_timeout_ms (int): Maximum time (milliseconds) a submitted
///         batch can remain pending. When exceeded, the operator enters a
///         terminal failed state. Default: ``60000``.
#[pyclass(name = "NvInferBatchingOperatorConfig", module = "savant_rs.nvinfer")]
pub struct PyNvInferBatchingOperatorConfig {
    pub(crate) inner: NvInferBatchingOperatorConfig,
}

#[pymethods]
impl PyNvInferBatchingOperatorConfig {
    #[new]
    #[pyo3(signature = (max_batch_size, max_batch_wait_ms, nvinfer_config, *, pending_batch_timeout_ms=60000))]
    fn new(
        max_batch_size: usize,
        max_batch_wait_ms: u64,
        nvinfer_config: &PyNvInferConfig,
        pending_batch_timeout_ms: u64,
    ) -> Self {
        Self {
            inner: NvInferBatchingOperatorConfig {
                max_batch_size,
                max_batch_wait: Duration::from_millis(max_batch_wait_ms),
                nvinfer: nvinfer_config.inner.clone(),
                pending_batch_timeout: Duration::from_millis(pending_batch_timeout_ms),
            },
        }
    }

    /// Maximum batch size; triggers inference when reached.
    #[getter]
    fn max_batch_size(&self) -> usize {
        self.inner.max_batch_size
    }

    /// Maximum wait before submitting a partial batch (milliseconds).
    #[getter]
    fn max_batch_wait_ms(&self) -> u64 {
        self.inner.max_batch_wait.as_millis() as u64
    }

    /// Pending batch timeout (milliseconds).
    #[getter]
    fn pending_batch_timeout_ms(&self) -> u64 {
        self.inner.pending_batch_timeout.as_millis() as u64
    }

    /// The embedded NvInfer engine configuration.
    #[getter]
    fn nvinfer_config(&self) -> PyNvInferConfig {
        PyNvInferConfig {
            inner: self.inner.nvinfer.clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "NvInferBatchingOperatorConfig(max_batch_size={}, \
             max_batch_wait_ms={}, gpu_id={})",
            self.inner.max_batch_size,
            self.inner.max_batch_wait.as_millis(),
            self.inner.nvinfer.gpu_id,
        )
    }
}

// ─── BatchFormationResult ──────────────────────────────────────────────

/// Result returned by the batch formation callback.
///
/// Args:
///     ids (list[tuple[SavantIdMetaKind, int]]): Per-frame Savant IDs.
///     rois (list[RoiKind]): Per-frame ROI specification.
#[pyclass(name = "BatchFormationResult", module = "savant_rs.nvinfer")]
pub struct PyBatchFormationResult {
    pub(super) inner: BatchFormationResult,
}

#[pymethods]
impl PyBatchFormationResult {
    #[new]
    fn new(ids: Vec<(PySavantIdMetaKind, u128)>, rois: Vec<PyRef<'_, PyRoiKind>>) -> Self {
        let rust_ids: Vec<SavantIdMetaKind> = ids
            .into_iter()
            .map(|(kind, id)| to_rust_id_kind(kind, id))
            .collect();
        let rust_rois: Vec<RoiKind> = rois.iter().map(|r| r.to_rust()).collect();
        Self {
            inner: BatchFormationResult {
                ids: rust_ids,
                rois: rust_rois,
            },
        }
    }

    /// Per-frame Savant IDs as ``(SavantIdMetaKind, int)`` tuples.
    #[getter]
    fn ids(&self) -> Vec<(PySavantIdMetaKind, u128)> {
        self.inner.ids.iter().map(from_rust_id_kind).collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "BatchFormationResult(ids_len={}, rois_len={})",
            self.inner.ids.len(),
            self.inner.rois.len(),
        )
    }
}

// ─── OperatorTensorView ─────────────────────────────────────────────────

/// Zero-copy view into a single output tensor from an operator slot.
///
/// Exposes ``host_ptr`` and ``device_ptr`` as plain integer addresses so
/// that Python callers can construct framework-native tensors (NumPy, CuPy,
/// PyTorch) without any data copy on the Rust side.
///
/// Valid while the parent ``OperatorInferenceOutput`` is alive.
#[pyclass(name = "OperatorTensorView", module = "savant_rs.nvinfer")]
pub struct PyOperatorTensorView {
    #[allow(dead_code)]
    shared: SharedOperatorOutput,
    name: String,
    dims: PyInferDims,
    data_type: PyDataType,
    byte_length: usize,
    host_ptr: usize,
    device_ptr: usize,
    has_host_data: bool,
}

#[pymethods]
impl PyOperatorTensorView {
    /// Output layer name.
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    /// Tensor dimensions.
    #[getter]
    fn dims(&self) -> PyInferDims {
        self.dims.clone()
    }

    /// Data type of tensor elements.
    #[getter]
    fn data_type(&self) -> PyDataType {
        self.data_type
    }

    /// Byte length of the tensor.
    #[getter]
    fn byte_length(&self) -> usize {
        self.byte_length
    }

    /// Host (CPU) memory address of the tensor data, or 0 if unavailable.
    #[getter]
    fn host_ptr(&self) -> usize {
        self.host_ptr
    }

    /// Device (GPU) memory address of the tensor data, or 0 if unavailable.
    #[getter]
    fn device_ptr(&self) -> usize {
        self.device_ptr
    }

    /// Whether host (CPU) tensor data is valid.
    ///
    /// Returns ``False`` when ``disable_output_host_copy`` was set on the
    /// config, meaning only ``device_ptr`` is usable.
    #[getter]
    fn has_host_data(&self) -> bool {
        self.has_host_data
    }

    /// NumPy-compatible dtype string (``"float32"``, ``"float16"``,
    /// ``"int8"``, ``"int32"``).
    #[getter]
    fn numpy_dtype(&self) -> &'static str {
        match self.data_type {
            PyDataType::Float => "float32",
            PyDataType::Half => "float16",
            PyDataType::Int8 => "int8",
            PyDataType::Int32 => "int32",
        }
    }

    /// Return tensor data as a NumPy array (zero-copy view).
    ///
    /// The returned array shares memory with the inference output buffer;
    /// it is valid as long as the parent ``OperatorInferenceOutput`` is alive.
    ///
    /// Raises:
    ///     RuntimeError: If host data is unavailable (``has_host_data`` is
    ///         ``False``) or the host pointer is null.
    fn as_numpy<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        super::output::tensor_as_numpy(
            py,
            self.has_host_data,
            self.host_ptr,
            self.byte_length,
            self.data_type,
            &self.dims.dims,
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "OperatorTensorView(name={:?}, dims={:?}, data_type={}, byte_length={}, \
             host_ptr=0x{:x}, device_ptr=0x{:x}, has_host_data={})",
            self.name,
            self.dims.dims,
            self.data_type.repr_str(),
            self.byte_length,
            self.host_ptr,
            self.device_ptr,
            self.has_host_data,
        )
    }
}

// ─── OperatorElementOutput ──────────────────────────────────────────────

/// Per-element inference output for one ROI in one operator frame.
///
/// Tensor pointers are valid while the parent ``OperatorInferenceOutput`` (or
/// any sibling ``OperatorTensorView``) is alive.
#[pyclass(name = "OperatorElementOutput", module = "savant_rs.nvinfer")]
pub struct PyOperatorElementOutput {
    shared: SharedOperatorOutput,
    slot_idx: usize,
    element_idx: usize,
    roi_id: Option<i64>,
    slot_number: u32,
    num_tensors: usize,
}

impl PyOperatorElementOutput {
    fn get_coordinate_scaler(&self) -> PyResult<nvinfer::CoordinateScaler> {
        let guard = self.shared.lock();
        let output = guard.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("OperatorInferenceOutput has been released")
        })?;
        let elem = &output.frames()[self.slot_idx].elements[self.element_idx];
        Ok(elem.coordinate_scaler())
    }
}

#[pymethods]
impl PyOperatorElementOutput {
    /// ROI identifier from ``Roi.id``.  ``None`` when no explicit ROIs were
    /// supplied and the full frame was used.
    #[getter]
    fn roi_id(&self) -> Option<i64> {
        self.roi_id
    }

    /// DeepStream surface slot index (``NvDsFrameMeta.batch_id``).
    #[getter]
    fn slot_number(&self) -> u32 {
        self.slot_number
    }

    /// Output tensors for this element.
    #[getter]
    fn tensors(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let guard = self.shared.lock();
        let output = guard.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("OperatorInferenceOutput has been released")
        })?;
        let elem = &output.frames()[self.slot_idx].elements[self.element_idx];
        let list = PyList::empty(py);
        for tv in elem.tensors.iter() {
            let py_tv = PyOperatorTensorView {
                shared: self.shared.clone(),
                name: tv.name.clone(),
                dims: PyInferDims {
                    dims: tv.dims.dimensions.clone(),
                    num_elements: tv.dims.num_elements,
                },
                data_type: tv.data_type.into(),
                byte_length: tv.byte_length,
                host_ptr: tv.host_ptr as usize,
                device_ptr: tv.device_ptr as usize,
                has_host_data: tv.host_copy_enabled,
            };
            list.append(Py::new(py, py_tv)?)?;
        }
        Ok(list.unbind())
    }

    /// Transform points from model-input space to absolute frame coordinates.
    ///
    /// Args:
    ///     data: ``ndarray[float32]`` of shape ``(N, 2)`` **or**
    ///         ``list[tuple[float, float]]`` of ``(x, y)`` points.
    ///
    /// Returns:
    ///     ``ndarray[float32]`` of shape ``(N, 2)``.
    fn scale_points<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let scaler = self.get_coordinate_scaler()?;
        let points = extract_f32x2(data)?;
        let result = py.detach(|| scaler.scale_points(&points));
        Ok(points_to_array(py, &result))
    }

    /// Transform axis-aligned boxes ``(left, top, width, height)`` from
    /// model-input space to absolute frame coordinates.
    ///
    /// Args:
    ///     data: ``ndarray[float32]`` of shape ``(N, 4)`` **or**
    ///         ``list[tuple[float, float, float, float]]``.
    ///
    /// Returns:
    ///     ``ndarray[float32]`` of shape ``(N, 4)``.
    fn scale_ltwh<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let scaler = self.get_coordinate_scaler()?;
        let boxes = extract_f32x4(data)?;
        let result = py.detach(|| scaler.scale_ltwh_batch(&boxes));
        Ok(f32x4_to_array(py, &result))
    }

    /// Transform axis-aligned boxes ``(left, top, right, bottom)`` from
    /// model-input space to absolute frame coordinates.
    ///
    /// Args:
    ///     data: ``ndarray[float32]`` of shape ``(N, 4)`` **or**
    ///         ``list[tuple[float, float, float, float]]``.
    ///
    /// Returns:
    ///     ``ndarray[float32]`` of shape ``(N, 4)``.
    fn scale_ltrb<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let scaler = self.get_coordinate_scaler()?;
        let boxes = extract_f32x4(data)?;
        let result = py.detach(|| scaler.scale_ltrb_batch(&boxes));
        Ok(f32x4_to_array(py, &result))
    }

    /// Transform rotated bounding boxes from model-input space to absolute
    /// frame coordinates.
    ///
    /// Args:
    ///     boxes: ``list[RBBox]``.
    ///
    /// Returns:
    ///     ``list[RBBox]``.
    fn scale_rbboxes<'py>(
        &self,
        py: Python<'py>,
        boxes: Vec<PyRef<'py, PyRBBox>>,
    ) -> PyResult<Vec<PyRBBox>> {
        let scaler = self.get_coordinate_scaler()?;
        let rust_boxes: Vec<savant_core::primitives::RBBox> =
            boxes.iter().map(|b| b.0.clone()).collect();
        let result = py.detach(|| scaler.scale_rbboxes(&rust_boxes));
        Ok(result.into_iter().map(PyRBBox).collect())
    }

    fn __repr__(&self) -> String {
        format!(
            "OperatorElementOutput(roi_id={:?}, slot_number={}, num_tensors={})",
            self.roi_id, self.slot_number, self.num_tensors,
        )
    }
}

/// Extract ``(N, 2)`` float data from either a numpy ndarray or a Python list.
fn extract_f32x2(data: &Bound<'_, PyAny>) -> PyResult<Vec<(f32, f32)>> {
    if let Ok(arr) = data.extract::<PyReadonlyArray2<f32>>() {
        let view = arr.as_array();
        if view.ncols() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Expected ndarray with shape (N, 2)",
            ));
        }
        Ok(view.rows().into_iter().map(|r| (r[0], r[1])).collect())
    } else {
        data.extract()
    }
}

/// Extract ``(N, 4)`` float data from either a numpy ndarray or a Python list.
fn extract_f32x4(data: &Bound<'_, PyAny>) -> PyResult<Vec<[f32; 4]>> {
    if let Ok(arr) = data.extract::<PyReadonlyArray2<f32>>() {
        let view = arr.as_array();
        if view.ncols() != 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Expected ndarray with shape (N, 4)",
            ));
        }
        Ok(view
            .rows()
            .into_iter()
            .map(|r| [r[0], r[1], r[2], r[3]])
            .collect())
    } else {
        let tuples: Vec<(f32, f32, f32, f32)> = data.extract()?;
        Ok(tuples
            .into_iter()
            .map(|(a, b, c, d)| [a, b, c, d])
            .collect())
    }
}

fn points_to_array<'py>(py: Python<'py>, pts: &[(f32, f32)]) -> Bound<'py, PyArray2<f32>> {
    let flat: Vec<f32> = pts.iter().flat_map(|&(x, y)| [x, y]).collect();
    PyArray2::from_vec2(py, &flat.chunks(2).map(|c| c.to_vec()).collect::<Vec<_>>()).unwrap()
}

fn f32x4_to_array<'py>(py: Python<'py>, boxes: &[[f32; 4]]) -> Bound<'py, PyArray2<f32>> {
    let rows: Vec<Vec<f32>> = boxes.iter().map(|b| b.to_vec()).collect();
    PyArray2::from_vec2(py, &rows).unwrap()
}

// ─── OperatorFrameOutput ────────────────────────────────────────────────

/// Per-frame inference result (callback view — no buffer access).
///
/// The per-frame buffer is held internally and only accessible after
/// calling ``OperatorInferenceOutput.take_deliveries()`` and then
/// ``SealedDeliveries.unseal()``.
///
/// Tensor pointers in ``elements`` are valid while the parent
/// ``OperatorInferenceOutput`` (or any derived child object) is alive.
#[pyclass(name = "OperatorFrameOutput", module = "savant_rs.nvinfer")]
pub struct PyOperatorFrameOutput {
    shared: SharedOperatorOutput,
    slot_idx: usize,
    frame: savant_core::primitives::frame::VideoFrameProxy,
    num_elements: usize,
}

#[pymethods]
impl PyOperatorFrameOutput {
    /// The original ``VideoFrame`` submitted for this frame.
    #[getter]
    fn frame(&self) -> VideoFrame {
        VideoFrame(self.frame.clone())
    }

    /// Inference results for this frame (one per ROI).
    #[getter]
    fn elements(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let guard = self.shared.lock();
        let output = guard.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("OperatorInferenceOutput has been released")
        })?;
        let frame_output = &output.frames()[self.slot_idx];
        let list = PyList::empty(py);
        for (eidx, elem) in frame_output.elements.iter().enumerate() {
            let py_elem = PyOperatorElementOutput {
                shared: self.shared.clone(),
                slot_idx: self.slot_idx,
                element_idx: eidx,
                roi_id: elem.roi_id,
                slot_number: elem.slot_number,
                num_tensors: elem.tensors.len(),
            };
            list.append(Py::new(py, py_elem)?)?;
        }
        Ok(list.unbind())
    }

    fn __repr__(&self) -> String {
        format!(
            "OperatorFrameOutput(source={:?}, num_elements={})",
            self.frame.get_source_id(),
            self.num_elements,
        )
    }
}

// ─── SealedDeliveries ───────────────────────────────────────────────────

/// A batch of ``(VideoFrame, SharedBuffer)`` pairs sealed until the
/// associated ``OperatorInferenceOutput`` is dropped.
///
/// Individual buffers are inaccessible while sealed.  Call
/// :meth:`unseal` (blocking) or :meth:`try_unseal` (non-blocking)
/// to obtain the pairs.
///
/// **Drop safety**: dropping ``SealedDeliveries`` without calling
/// ``unseal()`` is safe — contained buffers are freed and no deadlock
/// can occur.
#[pyclass(name = "SealedDeliveries", module = "savant_rs.nvinfer")]
pub struct PySealedDeliveries {
    inner: Option<SealedDeliveries>,
}

impl PySealedDeliveries {
    pub(crate) fn from_rust(sealed: SealedDeliveries) -> Self {
        Self {
            inner: Some(sealed),
        }
    }

    fn ensure_alive(&self) -> PyResult<&SealedDeliveries> {
        self.inner.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("SealedDeliveries already consumed")
        })
    }
}

#[pymethods]
impl PySealedDeliveries {
    /// Number of frames in the sealed batch.
    fn __len__(&self) -> PyResult<usize> {
        Ok(self.ensure_alive()?.len())
    }

    /// Whether the batch is empty.
    fn is_empty(&self) -> PyResult<bool> {
        Ok(self.ensure_alive()?.is_empty())
    }

    /// Whether the seal has been released (non-blocking check).
    ///
    /// Returns ``True`` once the ``OperatorInferenceOutput`` has been
    /// dropped.
    fn is_released(&self) -> PyResult<bool> {
        Ok(self.ensure_alive()?.is_released())
    }

    /// Block until the ``OperatorInferenceOutput`` is dropped, then
    /// return all deliveries as ``list[tuple[VideoFrame, SharedBuffer]]``.
    ///
    /// The GIL is released during the blocking wait so the callback
    /// thread (which needs the GIL to drop the output) can proceed.
    ///
    /// Args:
    ///     timeout_ms: Optional timeout in milliseconds.  When ``None``
    ///         (default), blocks indefinitely.  When the timeout expires,
    ///         raises ``TimeoutError``.
    ///
    /// Raises:
    ///     RuntimeError: If already consumed by a previous call.
    ///     TimeoutError: If the timeout expires before the seal is released.
    #[pyo3(signature = (timeout_ms=None))]
    fn unseal(&mut self, py: Python<'_>, timeout_ms: Option<u64>) -> PyResult<Py<PyList>> {
        let sealed = self.inner.take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("SealedDeliveries already consumed")
        })?;
        let pairs = match timeout_ms {
            Some(ms) => {
                let timeout = Duration::from_millis(ms);
                match py.detach(move || sealed.unseal_timeout(timeout)) {
                    Ok(pairs) => pairs,
                    Err(still_sealed) => {
                        self.inner = Some(still_sealed);
                        return Err(pyo3::exceptions::PyTimeoutError::new_err(
                            "unseal timed out",
                        ));
                    }
                }
            }
            None => py.detach(move || sealed.unseal()),
        };
        let list = PyList::empty(py);
        for (frame, buffer) in pairs {
            let py_frame = VideoFrame(frame);
            let py_buf = PySharedBuffer::from_rust(buffer);
            list.append((py_frame.into_pyobject(py)?, py_buf.into_pyobject(py)?))?;
        }
        Ok(list.unbind())
    }

    /// Non-blocking attempt to unseal.
    ///
    /// Returns ``list[tuple[VideoFrame, SharedBuffer]]`` if the seal
    /// has been released, or ``None`` if still sealed.
    ///
    /// Raises:
    ///     RuntimeError: If already consumed by a previous call.
    fn try_unseal(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyList>>> {
        let sealed = self.inner.take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("SealedDeliveries already consumed")
        })?;
        match sealed.try_unseal() {
            Ok(pairs) => {
                let list = PyList::empty(py);
                for (frame, buffer) in pairs {
                    let py_frame = VideoFrame(frame);
                    let py_buf = PySharedBuffer::from_rust(buffer);
                    list.append((py_frame.into_pyobject(py)?, py_buf.into_pyobject(py)?))?;
                }
                Ok(Some(list.unbind()))
            }
            Err(still_sealed) => {
                self.inner = Some(still_sealed);
                Ok(None)
            }
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            Some(s) => format!(
                "SealedDeliveries(len={}, released={})",
                s.len(),
                s.is_released()
            ),
            None => "SealedDeliveries(consumed)".to_string(),
        }
    }
}

// ─── OperatorInferenceOutput ────────────────────────────────────────────

/// Full batch inference result from the batching operator.
///
/// Owns the GStreamer buffer that backs all tensor pointers.  Data remains
/// valid as long as this object (or any child ``OperatorTensorView``) is alive.
///
/// Call :meth:`take_deliveries` to extract a :class:`SealedDeliveries`
/// containing the ``(VideoFrame, SharedBuffer)`` pairs for downstream.
#[pyclass(name = "OperatorInferenceOutput", module = "savant_rs.nvinfer")]
pub struct PyOperatorInferenceOutput {
    shared: SharedOperatorOutput,
    num_frames: usize,
    host_copy_enabled: bool,
}

impl PyOperatorInferenceOutput {
    pub(crate) fn from_rust(output: OperatorInferenceOutput) -> Self {
        let num_frames = output.frames().len();
        let host_copy_enabled = output.host_copy_enabled();
        Self {
            shared: Arc::new(Mutex::new(Some(output))),
            num_frames,
            host_copy_enabled,
        }
    }

    /// Second handle sharing the same underlying operator output (same ``Arc``).
    pub(crate) fn share(&self) -> Self {
        Self {
            shared: self.shared.clone(),
            num_frames: self.num_frames,
            host_copy_enabled: self.host_copy_enabled,
        }
    }
}

#[pymethods]
impl PyOperatorInferenceOutput {
    /// Per-frame outputs (inference results only — no buffer access).
    #[getter]
    fn frames(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let guard = self.shared.lock();
        let output = guard.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("OperatorInferenceOutput has been released")
        })?;
        let list = PyList::empty(py);
        for (idx, frame_output) in output.frames().iter().enumerate() {
            let py_frame = PyOperatorFrameOutput {
                shared: self.shared.clone(),
                slot_idx: idx,
                frame: frame_output.frame.clone(),
                num_elements: frame_output.elements.len(),
            };
            list.append(Py::new(py, py_frame)?)?;
        }
        Ok(list.unbind())
    }

    /// Whether host (CPU) tensor buffers contain valid data.
    #[getter]
    fn host_copy_enabled(&self) -> bool {
        self.host_copy_enabled
    }

    /// Number of frames in the batch.
    #[getter]
    fn num_frames(&self) -> usize {
        self.num_frames
    }

    /// Extract sealed deliveries while keeping tensor data alive.
    ///
    /// Returns a :class:`SealedDeliveries` on the first call.
    /// Subsequent calls return ``None``.
    fn take_deliveries(&self) -> PyResult<Option<PySealedDeliveries>> {
        let mut guard = self.shared.lock();
        let output = guard.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("OperatorInferenceOutput has been released")
        })?;
        Ok(output.take_deliveries().map(PySealedDeliveries::from_rust))
    }

    fn __repr__(&self) -> String {
        format!(
            "OperatorInferenceOutput(num_frames={}, host_copy_enabled={})",
            self.num_frames, self.host_copy_enabled,
        )
    }
}

// ─── OperatorOutput (discriminated callback payload) ─────────────────────

enum PyOperatorOutputInner {
    Inference(PyOperatorInferenceOutput),
    Eos { source_id: String },
    Error { message: String },
}

/// Discriminated result from [`NvInferBatchingOperator`]: inference batch, per-source EOS, or error.
///
/// Use ``is_inference`` / ``is_eos`` / ``is_error`` and ``as_operator_inference_output()``,
/// ``eos_source_id``, ``error_message`` as appropriate.
#[pyclass(name = "OperatorOutput", module = "savant_rs.nvinfer")]
pub struct PyOperatorOutput {
    inner: PyOperatorOutputInner,
}

impl PyOperatorOutput {
    pub(crate) fn from_rust(output: OperatorOutput) -> Self {
        let inner = match output {
            OperatorOutput::Inference(o) => {
                PyOperatorOutputInner::Inference(PyOperatorInferenceOutput::from_rust(o))
            }
            OperatorOutput::Eos { source_id } => PyOperatorOutputInner::Eos { source_id },
            OperatorOutput::Error(e) => PyOperatorOutputInner::Error {
                message: e.to_string(),
            },
        };
        Self { inner }
    }
}

#[pymethods]
impl PyOperatorOutput {
    #[getter]
    fn is_inference(&self) -> bool {
        matches!(self.inner, PyOperatorOutputInner::Inference(_))
    }

    #[getter]
    fn is_eos(&self) -> bool {
        matches!(self.inner, PyOperatorOutputInner::Eos { .. })
    }

    #[getter]
    fn is_error(&self) -> bool {
        matches!(self.inner, PyOperatorOutputInner::Error { .. })
    }

    /// Inference payload if this is an inference result, else ``None``.
    fn as_operator_inference_output(&self) -> Option<PyOperatorInferenceOutput> {
        match &self.inner {
            PyOperatorOutputInner::Inference(o) => Some(o.share()),
            _ => None,
        }
    }

    #[getter]
    fn eos_source_id(&self) -> Option<String> {
        match &self.inner {
            PyOperatorOutputInner::Eos { source_id } => Some(source_id.clone()),
            _ => None,
        }
    }

    #[getter]
    fn error_message(&self) -> Option<String> {
        match &self.inner {
            PyOperatorOutputInner::Error { message } => Some(message.clone()),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            PyOperatorOutputInner::Inference(o) => {
                format!("OperatorOutput(Inference(num_frames={}))", o.num_frames)
            }
            PyOperatorOutputInner::Eos { source_id } => {
                format!("OperatorOutput(Eos(source_id={source_id:?}))")
            }
            PyOperatorOutputInner::Error { message } => {
                format!("OperatorOutput(Error({message:?}))")
            }
        }
    }
}

// ─── NvInferBatchingOperator ────────────────────────────────────────────────────

/// Higher-level batching layer over ``NvInfer``.
///
/// Accepts individual ``(VideoFrame, SharedBuffer)`` pairs, accumulates them
/// into batches according to configurable policies, and delivers per-frame
/// results via the ``result_callback``.
///
/// Args:
///     config (NvInferBatchingOperatorConfig): Batching policy and NvInfer engine
///         configuration (embedded in
///         ``NvInferBatchingOperatorConfig.nvinfer_config``).
///     batch_formation_callback (Callable[[list[VideoFrame]], BatchFormationResult]):
///         Called when a batch is ready; must return per-frame IDs and ROIs.
///     result_callback (Callable[[OperatorOutput], None]):
///         Called for each operator result: inference batch, per-source EOS, or pipeline error.
#[pyclass(name = "NvInferBatchingOperator", module = "savant_rs.nvinfer")]
pub struct PyNvInferBatchingOperator {
    inner: Option<NvInferBatchingOperator>,
}

#[pymethods]
impl PyNvInferBatchingOperator {
    #[new]
    fn new(
        py: Python<'_>,
        config: &PyNvInferBatchingOperatorConfig,
        batch_formation_callback: Py<PyAny>,
        result_callback: Py<PyAny>,
    ) -> PyResult<Self> {
        let rust_config = config.inner.clone();

        let batch_cb: BatchFormationCallback = Arc::new(move |frames| {
            Python::attach(|py| {
                let py_frames: Vec<VideoFrame> =
                    frames.iter().map(|f| VideoFrame(f.clone())).collect();

                let result = match batch_formation_callback.call1(py, (py_frames,)) {
                    Ok(r) => r,
                    Err(e) => {
                        log::error!("batch_formation_callback raised: {e}");
                        return BatchFormationResult {
                            ids: vec![],
                            rois: vec![],
                        };
                    }
                };

                let bound = result.bind(py);
                match bound.extract::<PyRef<'_, PyBatchFormationResult>>() {
                    Ok(py_result) => BatchFormationResult {
                        ids: py_result.inner.ids.clone(),
                        rois: py_result.inner.rois.clone(),
                    },
                    Err(e) => {
                        log::error!(
                            "batch_formation_callback must return BatchFormationResult: {e}"
                        );
                        BatchFormationResult {
                            ids: vec![],
                            rois: vec![],
                        }
                    }
                }
            })
        });

        let result_cb: nvinfer::OperatorResultCallback = Box::new(move |output: OperatorOutput| {
            Python::attach(|py| {
                let py_output = PyOperatorOutput::from_rust(output);
                if let Err(e) = result_callback.call1(py, (py_output,)) {
                    log::error!("NvInferBatchingOperator result_callback error: {e}");
                }
            });
        });

        let operator = py.detach(|| {
            NvInferBatchingOperator::new(rust_config, batch_cb, result_cb)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })?;

        Ok(Self {
            inner: Some(operator),
        })
    }

    /// Add a single frame for batched inference.
    ///
    /// The buffer is **consumed** if a ``SharedBuffer`` is passed.
    ///
    /// Args:
    ///     frame (VideoFrame): The video frame.
    ///     buffer (Union[SharedBuffer, int]): Frame buffer.
    ///
    /// Raises:
    ///     RuntimeError: If the operator is shut down or submission fails.
    fn add_frame(
        &self,
        py: Python<'_>,
        frame: &VideoFrame,
        buffer: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let op = self.inner.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("NvInferBatchingOperator is shut down")
        })?;
        let shared_buf = take_shared_buffer(buffer)?;
        let proxy = frame.0.clone();
        py.detach(|| {
            op.add_frame(proxy, shared_buf)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Submit the current partial batch immediately (if non-empty).
    ///
    /// Raises:
    ///     RuntimeError: If the operator is shut down or submission fails.
    fn flush(&self, py: Python<'_>) -> PyResult<()> {
        let op = self.inner.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("NvInferBatchingOperator is shut down")
        })?;
        py.detach(|| {
            op.flush()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Send logical per-source EOS to the inner NvInfer pipeline.
    fn send_eos(&self, py: Python<'_>, source_id: &str) -> PyResult<()> {
        let op = self.inner.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("NvInferBatchingOperator is shut down")
        })?;
        py.detach(|| {
            op.send_eos(source_id)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Flush pending frames, stop the timer thread, and shut down NvInfer.
    ///
    /// Raises:
    ///     RuntimeError: If the operator has already been shut down.
    fn shutdown(&mut self, py: Python<'_>) -> PyResult<()> {
        let mut op = self.inner.take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "NvInferBatchingOperator is already shut down",
            )
        })?;
        py.detach(|| {
            op.shutdown()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    fn __repr__(&self) -> &'static str {
        if self.inner.is_some() {
            "NvInferBatchingOperator(running)"
        } else {
            "NvInferBatchingOperator(shut_down)"
        }
    }
}
