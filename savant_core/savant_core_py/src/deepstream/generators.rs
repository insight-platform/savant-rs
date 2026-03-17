//! Buffer generators and batched surface types.

use super::buffer::{extract_buf_ptr, PyDsNvBufSurfaceGstBuffer};
use super::config::{PyRect, PyTransformConfig};
use super::enums::{extract_mem_type, extract_video_format, PyVideoFormat};
use deepstream_nvbufsurface::{
    DsNvNonUniformSurfaceBuffer, DsNvSurfaceBufferGenerator, DsNvUniformSurfaceBuffer,
    DsNvUniformSurfaceBufferGenerator, NvBufSurfaceMemType,
};
use gstreamer as gst;
use numpy::PyReadonlyArray3;
use pyo3::prelude::*;

// ─── DsNvSurfaceBufferGenerator ─────────────────────────────────────────

/// Python wrapper for DsNvSurfaceBufferGenerator.
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
#[pyclass(name = "DsNvSurfaceBufferGenerator", module = "savant_rs.deepstream")]
pub struct PyDsNvSurfaceBufferGenerator {
    inner: DsNvSurfaceBufferGenerator,
}

#[pymethods]
impl PyDsNvSurfaceBufferGenerator {
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

        let inner = DsNvSurfaceBufferGenerator::builder(format, width, height)
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
        self.inner.nvmm_caps().to_string()
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
    ///     GstBuffer: Guard owning the acquired buffer.
    #[pyo3(signature = (id=None))]
    fn acquire_surface(
        &self,
        py: Python<'_>,
        id: Option<i64>,
    ) -> PyResult<PyDsNvBufSurfaceGstBuffer> {
        let buffer = py.detach(|| {
            self.inner
                .acquire_surface(id)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })?;
        Ok(PyDsNvBufSurfaceGstBuffer::new(buffer))
    }

    /// Acquire a buffer and stamp PTS and duration on it.
    ///
    /// Convenience wrapper around :meth:`acquire_surface` that stamps
    /// PTS and duration on the buffer.
    ///
    /// Args:
    ///     pts_ns (int): Presentation timestamp in nanoseconds.
    ///     duration_ns (int): Frame duration in nanoseconds.
    ///     id (int or None): Optional buffer ID / frame index.
    ///
    /// Returns:
    ///     GstBuffer: Guard owning the acquired buffer.
    #[pyo3(signature = (pts_ns, duration_ns, id=None))]
    fn acquire_surface_with_params(
        &self,
        py: Python<'_>,
        pts_ns: u64,
        duration_ns: u64,
        id: Option<i64>,
    ) -> PyResult<PyDsNvBufSurfaceGstBuffer> {
        let buffer = py.detach(|| -> PyResult<gst::Buffer> {
            let mut buffer = self
                .inner
                .acquire_surface(id)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            {
                let buf_ref = buffer.make_mut();
                buf_ref.set_pts(gst::ClockTime::from_nseconds(pts_ns));
                buf_ref.set_duration(gst::ClockTime::from_nseconds(duration_ns));
            }
            Ok(buffer)
        })?;
        Ok(PyDsNvBufSurfaceGstBuffer::new(buffer))
    }

    /// Acquire a buffer and return ``(GstBuffer, data_ptr, pitch)``.
    #[pyo3(signature = (id=None))]
    fn acquire_surface_with_ptr(
        &self,
        py: Python<'_>,
        id: Option<i64>,
    ) -> PyResult<(PyDsNvBufSurfaceGstBuffer, usize, u32)> {
        let (buffer, data_ptr, pitch) = py.detach(|| -> PyResult<(gst::Buffer, usize, u32)> {
            let (buf, ptr, p) = self
                .inner
                .acquire_buffer(id)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok((buf, ptr as usize, p))
        })?;
        Ok((PyDsNvBufSurfaceGstBuffer::new(buffer), data_ptr, pitch))
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
    ) -> PyResult<PyDsNvBufSurfaceGstBuffer> {
        let src_buf_ptr = extract_buf_ptr(src_buf)?;
        let config = config.to_rust();
        let src_rect_rust = src_rect.map(|r| r.into_rust());
        let dst_buf = py.detach(|| {
            let src_buf =
                unsafe { gst::Buffer::from_glib_none(src_buf_ptr as *const gst::ffi::GstBuffer) };
            let src_view = SurfaceView::from_buffer(src_gst, 0)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            let dst_shared = self
                .inner
                .acquire_surface_with_ptr(id)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            {
                let dst_view = SurfaceView::from_shared(dst_shared.clone(), 0)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                src_view
                    .transform_into(&dst_view, &config, src_rect_rust.as_ref())
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            }
            dst_shared
                .into_buffer()
                .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("buffer still shared"))
        })?;
        Ok(PyDsNvBufSurfaceGstBuffer::new(dst_buf))
    }

    /// Like :meth:`transform` but also returns ``(GstBuffer, data_ptr, pitch)``.
    #[pyo3(signature = (src_buf, config, id=None, src_rect=None))]
    fn transform_with_ptr(
        &self,
        py: Python<'_>,
        src_buf: &Bound<'_, PyAny>,
        config: &PyTransformConfig,
        id: Option<i64>,
        src_rect: Option<&PyRect>,
    ) -> PyResult<(PyDsNvBufSurfaceGstBuffer, usize, u32)> {
        let src_buf_ptr = extract_buf_ptr(src_buf)?;
        let config = config.to_rust();
        let src_rect_rust = src_rect.map(|r| r.into_rust());
        let (dst_buf, data_ptr, pitch) = py.detach(|| -> PyResult<(gst::Buffer, usize, u32)> {
            let src_buf =
                unsafe { gst::Buffer::from_glib_none(src_buf_ptr as *const gst::ffi::GstBuffer) };
            let (buf, ptr, p) = self
                .inner
                .transform_with_ptr(&src_buf, &config, id, src_rect_rust.as_ref())
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok((buf, ptr as usize, p))
        })?;
        Ok((PyDsNvBufSurfaceGstBuffer::new(dst_buf), data_ptr, pitch))
    }

    /// Send an end-of-stream signal to an AppSrc element.
    #[staticmethod]
    fn send_eos(appsrc_ptr: usize) -> PyResult<()> {
        unsafe {
            DsNvSurfaceBufferGenerator::send_eos_raw(appsrc_ptr)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        }
    }

    /// Create a new NvBufSurface and attach it to the given buffer.
    #[pyo3(signature = (gst_buffer_dest, id=None))]
    fn create_surface(
        &self,
        py: Python<'_>,
        gst_buffer_dest: &Bound<'_, PyAny>,
        id: Option<i64>,
    ) -> PyResult<()> {
        let dest_ptr = extract_buf_ptr(gst_buffer_dest)?;
        py.detach(|| unsafe {
            self.inner
                .create_surface_raw(dest_ptr as *mut gst::ffi::GstBuffer, id)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }
}

// ─── DsNvUniformSurfaceBufferGenerator ───────────────────────────────────

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
#[pyclass(
    name = "DsNvUniformSurfaceBufferGenerator",
    module = "savant_rs.deepstream"
)]
pub struct PyDsNvUniformSurfaceBufferGenerator {
    inner: DsNvUniformSurfaceBufferGenerator,
}

#[pymethods]
impl PyDsNvUniformSurfaceBufferGenerator {
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

        let inner =
            DsNvUniformSurfaceBufferGenerator::builder(format, width, height, max_batch_size)
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

    /// Acquire a ``DsNvUniformSurfaceBuffer`` from the pool, ready for slot filling.
    fn acquire_batched_surface(
        &self,
        py: Python<'_>,
        config: &PyTransformConfig,
    ) -> PyResult<PyDsNvUniformSurfaceBuffer> {
        py.detach(|| {
            let batch = self
                .inner
                .acquire_batched_surface(config.to_rust())
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(PyDsNvUniformSurfaceBuffer { inner: batch })
        })
    }
}

// ─── DsNvUniformSurfaceBuffer ─────────────────────────────────────────────

/// Pool-allocated batched NvBufSurface with per-slot fill tracking.
///
/// Obtained from
/// ``DsNvUniformSurfaceBufferGenerator.acquire_batched_surface``.
/// Fill individual slots with ``fill_slot``, then call ``finalize``,
/// then ``as_gst_buffer`` to access the buffer.
#[pyclass(name = "DsNvUniformSurfaceBuffer", module = "savant_rs.deepstream")]
pub struct PyDsNvUniformSurfaceBuffer {
    inner: DsNvUniformSurfaceBuffer,
}

#[pymethods]
impl PyDsNvUniformSurfaceBuffer {
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

    /// Return ``(data_ptr, pitch)`` for a slot by index.
    fn slot_ptr(&self, index: u32) -> PyResult<(usize, u32)> {
        let (data_ptr, pitch) = self
            .inner
            .slot_ptr(index)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok((data_ptr as usize, pitch))
    }

    /// Transform a source buffer into the next available batch slot.
    #[pyo3(signature = (src_buf, src_rect=None, id=None))]
    fn fill_slot(
        &mut self,
        py: Python<'_>,
        src_buf: &Bound<'_, PyAny>,
        src_rect: Option<&PyRect>,
        id: Option<i64>,
    ) -> PyResult<()> {
        let src_buf_ptr = extract_buf_ptr(src_buf)?;
        let src_rect_rust = src_rect.map(|r| r.into_rust());
        py.detach(|| {
            let ptr = src_buf_ptr as *mut gst::ffi::GstBuffer;
            let src_buf = unsafe { gst::Buffer::from_glib_none(ptr) };
            self.inner
                .fill_slot(&src_buf, src_rect_rust.as_ref(), id)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Finalize the batch (non-consuming).
    fn finalize(&mut self) -> PyResult<()> {
        self.inner
            .finalize()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Return the underlying GstBuffer guard. Available only after ``finalize``.
    fn as_gst_buffer(&self) -> PyResult<PyDsNvBufSurfaceGstBuffer> {
        let buffer = self
            .inner
            .as_gst_buffer()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyDsNvBufSurfaceGstBuffer::new(buffer))
    }

    /// Create a zero-copy single-frame view of one filled slot.
    fn extract_slot_view(&self, slot_index: u32) -> PyResult<PyDsNvBufSurfaceGstBuffer> {
        let view = self
            .inner
            .extract_slot_view(slot_index)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyDsNvBufSurfaceGstBuffer::new(view))
    }

    /// Fill a slot's surface with a constant byte value.
    fn memset_slot(&self, py: Python<'_>, index: u32, value: u8) -> PyResult<()> {
        let slot_buf = self
            .inner
            .extract_slot_view(index)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        py.detach(|| {
            let view = deepstream_nvbufsurface::SurfaceView::from_buffer(slot_buf, 0)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            deepstream_nvbufsurface::memset_surface(&view, value)
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
        let slot_buf = self
            .inner
            .extract_slot_view(index)
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
            let view = deepstream_nvbufsurface::SurfaceView::from_buffer(slot_buf, 0)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            deepstream_nvbufsurface::upload_to_surface(&view, slice, width, height, channels)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }
}

// ─── DsNvNonUniformSurfaceBuffer ─────────────────────────────────────────

/// Zero-copy heterogeneous batch (nvstreammux2-style).
///
/// Assembles individual NvBufSurface buffers of arbitrary dimensions
/// and pixel formats into a single batched ``GstBuffer``.
///
/// Args:
///     max_batch_size (int): Maximum number of surfaces in the batch.
///     gpu_id (int): GPU device ID (default 0).
///
/// Raises:
///     RuntimeError: If batch creation fails.
#[pyclass(name = "DsNvNonUniformSurfaceBuffer", module = "savant_rs.deepstream")]
pub struct PyDsNvNonUniformSurfaceBuffer {
    inner: DsNvNonUniformSurfaceBuffer,
}

#[pymethods]
impl PyDsNvNonUniformSurfaceBuffer {
    #[new]
    #[pyo3(signature = (max_batch_size, gpu_id=0))]
    fn new(max_batch_size: u32, gpu_id: u32) -> PyResult<Self> {
        let inner = DsNvNonUniformSurfaceBuffer::new(max_batch_size, gpu_id)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    #[getter]
    fn num_filled(&self) -> u32 {
        self.inner.num_filled()
    }

    #[getter]
    fn max_batch_size(&self) -> u32 {
        self.inner.max_batch_size()
    }

    #[getter]
    fn gpu_id(&self) -> u32 {
        self.inner.gpu_id()
    }

    #[getter]
    fn is_finalized(&self) -> bool {
        self.inner.is_finalized()
    }

    /// Add a source buffer to the batch (zero-copy).
    #[pyo3(signature = (src_buf, id=None))]
    fn add(&mut self, src_buf: &Bound<'_, PyAny>, id: Option<i64>) -> PyResult<()> {
        let src_buf_ptr = extract_buf_ptr(src_buf)?;
        let src_buf =
            unsafe { gst::Buffer::from_glib_none(src_buf_ptr as *const gst::ffi::GstBuffer) };
        self.inner
            .add(&src_buf, id)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Return ``(data_ptr, pitch, width, height)`` for a slot by index.
    fn slot_ptr(&self, index: u32) -> PyResult<(usize, u32, u32, u32)> {
        let (data_ptr, pitch, width, height) = self
            .inner
            .slot_ptr(index)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok((data_ptr as usize, pitch, width, height))
    }

    /// Finalize the batch (non-consuming).
    fn finalize(&mut self) -> PyResult<()> {
        self.inner
            .finalize()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Return the underlying GstBuffer guard. Available only after ``finalize``.
    fn as_gst_buffer(&self) -> PyResult<PyDsNvBufSurfaceGstBuffer> {
        let buffer = self
            .inner
            .as_gst_buffer()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyDsNvBufSurfaceGstBuffer::new(buffer))
    }

    /// Create a zero-copy single-frame view of one filled slot.
    fn extract_slot_view(&self, slot_index: u32) -> PyResult<PyDsNvBufSurfaceGstBuffer> {
        let view = self
            .inner
            .extract_slot_view(slot_index)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyDsNvBufSurfaceGstBuffer::new(view))
    }
}
