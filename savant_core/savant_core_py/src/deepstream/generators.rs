//! Buffer generators and batched surface types.

use super::buffer::{extract_buf_ptr, PySharedBuffer};
use super::config::{PyRect, PyTransformConfig};
use super::enums::{
    extract_mem_type, extract_video_format, to_rust_id_kind, PySavantIdMetaKind, PyVideoFormat,
};
use super::surface_view::PySurfaceView;
use deepstream_buffers::{
    BufferGenerator, NonUniformBatch, NvBufSurfaceMemType, SharedBuffer, SurfaceBatch,
    UniformBatchGenerator,
};
use gstreamer as gst;
use numpy::PyReadonlyArray3;
use pyo3::prelude::*;

// ─── BufferGenerator ─────────────────────────────────────────

/// Python wrapper for BufferGenerator.
///
/// Args:
///     format (VideoFormat | str): Video format.
///     width (int): Frame width in pixels.
///     height (int): Frame height in pixels.
///     fps_num (int): Framerate numerator (default 30).
///     fps_den (int): Framerate denominator (default 1).
///     gpu_id (int): GPU device ID (default 0).
///     mem_type (MemType | int): Memory type (default ``MemType.DEFAULT``).
///     pool_size (int): Buffer pool size (default 4).
#[pyclass(name = "BufferGenerator", module = "savant_rs.deepstream")]
pub struct PyBufferGenerator {
    inner: BufferGenerator,
}

#[pymethods]
impl PyBufferGenerator {
    #[new]
    #[pyo3(signature = (format, width, height, fps_num=30, fps_den=1, gpu_id=0, mem_type=None, pool_size=4))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        format: &Bound<'_, PyAny>,
        width: u32,
        height: u32,
        fps_num: i32,
        fps_den: i32,
        gpu_id: u32,
        mem_type: Option<&Bound<'_, PyAny>>,
        pool_size: u32,
    ) -> PyResult<Self> {
        let format = extract_video_format(format)?;
        let mem_type = match mem_type {
            Some(m) => extract_mem_type(m)?,
            None => NvBufSurfaceMemType::Default,
        };

        let inner = BufferGenerator::builder(format, width, height)
            .fps(fps_num, fps_den)
            .gpu_id(gpu_id)
            .mem_type(mem_type)
            .min_buffers(pool_size)
            .max_buffers(pool_size)
            .build()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(Self { inner })
    }

    /// Return the NVMM caps string for configuring an ``appsrc``.
    fn nvmm_caps_str(&self) -> String {
        self.inner.nvmm_caps()
    }

    #[getter]
    fn width(&self) -> u32 {
        self.inner.width()
    }

    #[getter]
    fn height(&self) -> u32 {
        self.inner.height()
    }

    #[getter]
    fn format(&self) -> PyVideoFormat {
        self.inner.format().into()
    }

    /// Acquire a new NvBufSurface buffer from the pool.
    ///
    /// Returns:
    ///     SharedBuffer: Guard owning the acquired buffer.
    #[pyo3(signature = (id=None))]
    fn acquire(&self, py: Python<'_>, id: Option<i64>) -> PyResult<PySharedBuffer> {
        let shared = py.detach(|| {
            self.inner
                .acquire(id)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })?;
        Ok(PySharedBuffer::from_rust(shared))
    }

    /// Acquire a buffer and stamp PTS and duration on it.
    ///
    /// Convenience wrapper around :meth:`acquire` that stamps
    /// PTS and duration on the buffer.
    ///
    /// Args:
    ///     pts_ns (int): Presentation timestamp in nanoseconds.
    ///     duration_ns (int): Frame duration in nanoseconds.
    ///     id (int or None): Optional buffer ID / frame index.
    ///
    /// Returns:
    ///     SharedBuffer: Guard owning the acquired buffer.
    #[pyo3(signature = (pts_ns, duration_ns, id=None))]
    fn acquire_with_params(
        &self,
        py: Python<'_>,
        pts_ns: u64,
        duration_ns: u64,
        id: Option<i64>,
    ) -> PyResult<PySharedBuffer> {
        let shared = py.detach(|| -> PyResult<SharedBuffer> {
            let sb = self
                .inner
                .acquire(id)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            sb.set_pts_ns(pts_ns);
            sb.set_duration_ns(duration_ns);
            Ok(sb)
        })?;
        Ok(PySharedBuffer::from_rust(shared))
    }

    /// Transform (scale + letterbox) a source buffer into a new destination.
    #[pyo3(signature = (src_buf, config, id=None, src_rect=None))]
    fn transform(
        &self,
        py: Python<'_>,
        src_buf: &Bound<'_, PyAny>,
        config: &PyTransformConfig,
        id: Option<i64>,
        src_rect: Option<&PyRect>,
    ) -> PyResult<PySharedBuffer> {
        let src_buf_ptr = extract_buf_ptr(src_buf)?;
        let config = config.to_rust();
        let src_rect_rust = src_rect.map(|r| r.into_rust());
        let shared = py.detach(|| -> PyResult<SharedBuffer> {
            let src_buf =
                unsafe { gst::Buffer::from_glib_none(src_buf_ptr as *const gst::ffi::GstBuffer) };
            let src_shared = SharedBuffer::from(src_buf);
            let src_view = deepstream_buffers::SurfaceView::from_buffer(&src_shared, 0)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            let dst_shared = self
                .inner
                .acquire(id)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            {
                let dst_view = deepstream_buffers::SurfaceView::from_buffer(&dst_shared, 0)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                src_view
                    .transform_into(&dst_view, &config, src_rect_rust.as_ref())
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            }
            Ok(dst_shared)
        })?;
        Ok(PySharedBuffer::from_rust(shared))
    }

    /// Send an end-of-stream signal to an AppSrc element.
    #[staticmethod]
    fn send_eos(appsrc_ptr: usize) -> PyResult<()> {
        unsafe {
            deepstream_buffers::gst_app::send_eos_raw(appsrc_ptr)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        }
    }
}

// ─── UniformBatchGenerator ───────────────────────────────────

/// Homogeneous batched NvBufSurface buffer generator.
///
/// Produces buffers whose ``surfaceList`` is an array of independently
/// fillable GPU surfaces, all sharing the same pixel format and
/// dimensions.
///
/// Args:
///     format (VideoFormat | str): Pixel format (e.g. ``"RGBA"``).
///     width (int): Slot width in pixels.
///     height (int): Slot height in pixels.
///     max_batch_size (int): Maximum number of slots per batch.
///     pool_size (int): Number of pre-allocated batched buffers (default 2).
///     fps_num (int): Framerate numerator (default 30).
///     fps_den (int): Framerate denominator (default 1).
///     gpu_id (int): GPU device ID (default 0).
///     mem_type (MemType | None): Memory type (default ``MemType.DEFAULT``).
///
/// Raises:
///     RuntimeError: If pool creation fails.
#[pyclass(name = "UniformBatchGenerator", module = "savant_rs.deepstream")]
pub struct PyUniformBatchGenerator {
    inner: UniformBatchGenerator,
}

#[pymethods]
impl PyUniformBatchGenerator {
    #[new]
    #[pyo3(signature = (format, width, height, max_batch_size, pool_size=2, fps_num=30, fps_den=1, gpu_id=0, mem_type=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        format: &Bound<'_, PyAny>,
        width: u32,
        height: u32,
        max_batch_size: u32,
        pool_size: u32,
        fps_num: i32,
        fps_den: i32,
        gpu_id: u32,
        mem_type: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let format = extract_video_format(format)?;
        let mem_type = match mem_type {
            Some(m) => extract_mem_type(m)?,
            None => NvBufSurfaceMemType::Default,
        };

        let inner = UniformBatchGenerator::builder(format, width, height, max_batch_size)
            .fps(fps_num, fps_den)
            .gpu_id(gpu_id)
            .mem_type(mem_type)
            .pool_size(pool_size)
            .build()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(Self { inner })
    }

    #[getter]
    fn width(&self) -> u32 {
        self.inner.width()
    }

    #[getter]
    fn height(&self) -> u32 {
        self.inner.height()
    }

    #[getter]
    fn format(&self) -> PyVideoFormat {
        self.inner.format().into()
    }

    #[getter]
    fn gpu_id(&self) -> u32 {
        self.inner.gpu_id()
    }

    #[getter]
    fn max_batch_size(&self) -> u32 {
        self.inner.max_batch_size()
    }

    /// Acquire a ``SurfaceBatch`` from the pool, ready for slot filling.
    ///
    /// Args:
    ///     config (TransformConfig): Scaling / letterboxing configuration.
    ///     ids (list[tuple[SavantIdMetaKind, int]] | None): Optional per-slot
    ///         ``SavantIdMeta`` entries.
    #[pyo3(signature = (config, ids=None))]
    fn acquire_batch(
        &self,
        py: Python<'_>,
        config: &PyTransformConfig,
        ids: Option<Vec<(PySavantIdMetaKind, i64)>>,
    ) -> PyResult<PySurfaceBatch> {
        let id_kinds = ids
            .unwrap_or_default()
            .into_iter()
            .map(|(kind, id)| to_rust_id_kind(kind, id))
            .collect();
        py.detach(|| {
            let batch = self
                .inner
                .acquire_batch(config.to_rust(), id_kinds)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(PySurfaceBatch { inner: batch })
        })
    }
}

// ─── SurfaceBatch ─────────────────────────────────────────────

/// Pool-allocated batched NvBufSurface with per-slot fill tracking.
///
/// Obtained from
/// ``UniformBatchGenerator.acquire_batch``.
/// Fill individual slots with ``transform_slot``, then call ``finalize``,
/// then ``shared_buffer`` to access the buffer.
#[pyclass(name = "SurfaceBatch", module = "savant_rs.deepstream")]
pub struct PySurfaceBatch {
    inner: SurfaceBatch,
}

#[pymethods]
impl PySurfaceBatch {
    #[getter]
    fn num_filled(&self) -> u32 {
        self.inner.num_filled()
    }

    #[getter]
    fn max_batch_size(&self) -> u32 {
        self.inner.max_batch_size()
    }

    #[getter]
    fn is_finalized(&self) -> bool {
        self.inner.is_finalized()
    }

    /// Transform a source buffer into a specific batch slot.
    #[pyo3(signature = (slot, src_buf, src_rect=None))]
    fn transform_slot(
        &mut self,
        py: Python<'_>,
        slot: u32,
        src_buf: &Bound<'_, PyAny>,
        src_rect: Option<&PyRect>,
    ) -> PyResult<()> {
        let src_buf_ptr = extract_buf_ptr(src_buf)?;
        let src_rect_rust = src_rect.map(|r| r.into_rust());
        py.detach(|| {
            let ptr = src_buf_ptr as *mut gst::ffi::GstBuffer;
            let src_buf = unsafe { gst::Buffer::from_glib_none(ptr) };
            let src_shared = SharedBuffer::from(src_buf);
            let src_view = deepstream_buffers::SurfaceView::from_buffer(&src_shared, 0)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            self.inner
                .transform_slot(slot, &src_view, src_rect_rust.as_ref())
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Finalize the batch: set ``numFilled`` and attach IDs from acquisition.
    fn finalize(&mut self) -> PyResult<()> {
        self.inner
            .finalize()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Return the underlying ``SharedBuffer``. Available only after ``finalize``.
    fn shared_buffer(&self) -> PyResult<PySharedBuffer> {
        Ok(PySharedBuffer::from_rust(self.inner.shared_buffer()))
    }

    /// Create a zero-copy single-slot ``SurfaceView`` from the batch.
    fn view(&self, slot_index: u32) -> PyResult<PySurfaceView> {
        let view = self
            .inner
            .view(slot_index)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PySurfaceView::new(view))
    }

    /// Fill a slot's surface with a constant byte value.
    fn memset_slot(&self, py: Python<'_>, index: u32, value: u8) -> PyResult<()> {
        let view = self
            .inner
            .view(index)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        py.detach(|| {
            view.memset(value)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Upload pixel data from a NumPy array into a batch slot.
    fn upload_slot<'py>(
        &self,
        py: Python<'py>,
        index: u32,
        data: PyReadonlyArray3<'py, u8>,
    ) -> PyResult<()> {
        let view = self
            .inner
            .view(index)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let arr = data.as_array();
        let shape = arr.shape();
        let height = shape[0] as u32;
        let width = shape[1] as u32;
        let channels = shape[2] as u32;
        let slice = data.as_slice().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "array must be contiguous in memory: {e}"
            ))
        })?;
        py.detach(|| {
            view.upload(slice, width, height, channels)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }
}

// ─── NonUniformBatch ─────────────────────────────────────────

/// Zero-copy heterogeneous batch (nvstreammux2-style).
///
/// Assembles individual NvBufSurface buffers of arbitrary dimensions
/// and pixel formats into a single batched ``GstBuffer``.
///
/// Args:
///     gpu_id (int): GPU device ID (default 0).
#[pyclass(name = "NonUniformBatch", module = "savant_rs.deepstream", unsendable)]
pub struct PyNonUniformBatch {
    inner: Option<NonUniformBatch>,
    gpu_id: u32,
}

#[pymethods]
impl PyNonUniformBatch {
    #[new]
    #[pyo3(signature = (gpu_id=0))]
    fn new(gpu_id: u32) -> Self {
        Self {
            inner: Some(NonUniformBatch::new(gpu_id)),
            gpu_id,
        }
    }

    #[getter]
    fn num_filled(&self) -> PyResult<u32> {
        let batch = self
            .inner
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("batch has been finalized"))?;
        Ok(batch.num_filled())
    }

    #[getter]
    fn gpu_id(&self) -> u32 {
        self.gpu_id
    }

    /// Add a source ``SurfaceView`` to the batch (zero-copy).
    fn add(&mut self, src_view: &PySurfaceView) -> PyResult<()> {
        let batch = self
            .inner
            .as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("batch has been finalized"))?;
        let view_ref = src_view.inner_ref()?;
        batch
            .add(view_ref)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Finalize the batch and return the underlying ``SharedBuffer``.
    ///
    /// The batch is consumed; further calls will raise ``RuntimeError``.
    ///
    /// Args:
    ///     ids (list[tuple[SavantIdMetaKind, int]] | None): Optional per-slot
    ///         ``SavantIdMeta`` entries.
    #[pyo3(signature = (ids=None))]
    fn finalize(
        &mut self,
        ids: Option<Vec<(PySavantIdMetaKind, i64)>>,
    ) -> PyResult<PySharedBuffer> {
        let batch = self.inner.take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("batch has already been finalized")
        })?;
        let id_kinds = ids
            .unwrap_or_default()
            .into_iter()
            .map(|(kind, id)| to_rust_id_kind(kind, id))
            .collect();
        let shared = batch
            .finalize(id_kinds)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PySharedBuffer::from_rust(shared))
    }
}
