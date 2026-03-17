//! `DsNvBufSurfaceGstBuffer` — RAII guard for NvBufSurface-backed GStreamer
//! buffers, plus utility functions for extracting buffer pointers and shared
//! references from Python arguments.

use deepstream_buffers::SharedBuffer;
use glib::translate::from_glib_none;
use gstreamer as gst;
use pyo3::prelude::*;

// ─── DsNvBufSurfaceGstBuffer guard ──────────────────────────────────────

/// RAII guard for an NvBufSurface-backed ``GstBuffer``.
///
/// Wraps a GStreamer buffer and automatically unrefs it when the Python
/// object is garbage-collected.  Use ``ptr`` to obtain the raw pointer
/// for interop with functions that accept raw addresses, and ``take``
/// to transfer ownership out of the guard.
#[pyclass(name = "DsNvBufSurfaceGstBuffer", module = "savant_rs.deepstream")]
pub struct PyDsNvBufSurfaceGstBuffer {
    inner: Option<SharedBuffer>,
}

const CONSUMED_MSG: &str = "DsNvBufSurfaceGstBuffer has already been consumed via take(); \
                            create a new one with as_gst_buffer() or from_ptr()";

impl PyDsNvBufSurfaceGstBuffer {
    pub fn new(buffer: gst::Buffer) -> Self {
        Self {
            inner: Some(SharedBuffer::from(buffer)),
        }
    }

    /// Return the raw pointer as `usize`, or an error if consumed.
    pub fn ptr_usize(&self) -> PyResult<usize> {
        let shared = self
            .inner
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(CONSUMED_MSG))?;
        let guard = shared.lock();
        Ok(guard.as_ref().as_ptr() as usize)
    }

    /// Borrow the inner [`SharedBuffer`].
    ///
    /// Callers that need a [`SurfaceView`] should clone this (cheap Arc
    /// clone) and pass it to [`SurfaceView::from_buffer`].  The GstBuffer
    /// GLib refcount stays at 1, avoiding COW in `resolve_cuda_ptr` and
    /// preserving POOLED `EglCudaMeta` across pool recycles.
    pub(crate) fn shared(&self) -> PyResult<&SharedBuffer> {
        self.inner
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(CONSUMED_MSG))
    }

    /// Take exclusive ownership of the inner `gst::Buffer`, leaving the
    /// guard empty.  Fails if the Arc has outstanding clones (e.g. a
    /// `SurfaceView` still holds a reference).
    pub(crate) fn take_buffer(&mut self) -> PyResult<gst::Buffer> {
        let shared = self
            .inner
            .take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(CONSUMED_MSG))?;
        shared.into_buffer().map_err(|returned| {
            self.inner = Some(returned);
            pyo3::exceptions::PyRuntimeError::new_err(
                "buffer is shared with outstanding SurfaceViews; drop them first",
            )
        })
    }
}

#[pymethods]
impl PyDsNvBufSurfaceGstBuffer {
    /// Wrap a raw ``GstBuffer*`` pointer in a guard.
    ///
    /// Args:
    ///     ptr (int): Raw ``GstBuffer*`` pointer address.
    ///     add_ref (bool): If ``True`` (default) an additional reference
    ///         is taken — use for borrowed pointers (pad probes,
    ///         callbacks).  If ``False`` the guard assumes ownership of
    ///         an existing reference — use for pointers obtained via the
    ///         legacy ``int``-returning API.
    ///
    /// Raises:
    ///     ValueError: If *ptr* is 0 (null).
    #[staticmethod]
    #[pyo3(signature = (ptr, add_ref=true))]
    fn from_ptr(ptr: usize, add_ref: bool) -> PyResult<Self> {
        if ptr == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("ptr is null"));
        }
        let buffer = unsafe {
            if add_ref {
                gst::Buffer::from_glib_none(ptr as *const gst::ffi::GstBuffer)
            } else {
                gst::Buffer::from_glib_full(ptr as *mut gst::ffi::GstBuffer)
            }
        };
        Ok(Self::new(buffer))
    }

    /// Raw ``GstBuffer*`` pointer address.
    ///
    /// Raises:
    ///     RuntimeError: If the buffer has been consumed via ``take``.
    #[getter]
    fn ptr(&self) -> PyResult<usize> {
        self.ptr_usize()
    }

    /// Transfer ownership out of the guard and return the raw pointer.
    ///
    /// After this call the guard is empty — ``ptr`` will raise and the
    /// destructor becomes a no-op.
    ///
    /// Returns:
    ///     int: Raw ``GstBuffer*`` pointer (caller owns the reference).
    ///
    /// Raises:
    ///     RuntimeError: If already consumed or buffer is shared.
    fn take(&mut self) -> PyResult<usize> {
        let buffer = self.take_buffer()?;
        let raw = unsafe {
            use glib::translate::IntoGlibPtr;
            buffer.into_glib_ptr() as usize
        };
        Ok(raw)
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            Some(shared) => {
                let guard = shared.lock();
                format!(
                    "DsNvBufSurfaceGstBuffer(ptr=0x{:x})",
                    guard.as_ref().as_ptr() as usize
                )
            }
            None => "DsNvBufSurfaceGstBuffer(<consumed>)".to_string(),
        }
    }

    fn __bool__(&self) -> bool {
        self.inner.is_some()
    }
}

// ─── Buffer extraction utilities ─────────────────────────────────────────

/// Obtain a `&mut gst::BufferRef` from a Python buffer argument.
///
/// When `buf` is a [`PyDsNvBufSurfaceGstBuffer`] the inner `gst::Buffer` is
/// accessed via the `SharedBuffer` mutex lock.  Because the
/// GstBuffer GLib refcount is always 1, `make_mut()` succeeds in-place
/// without COW.  For raw integer pointers writability is checked explicitly.
pub(crate) fn with_mut_buffer_ref<F, R>(buf: &Bound<'_, PyAny>, f: F) -> PyResult<R>
where
    F: FnOnce(&mut gst::BufferRef) -> PyResult<R>,
{
    if let Ok(guard) = buf.extract::<PyRef<'_, PyDsNvBufSurfaceGstBuffer>>() {
        let shared = guard.shared()?;
        let mut lock = shared.lock();
        let buf_ref = lock.make_mut();
        return f(buf_ref);
    }

    let raw = buf.extract::<usize>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "expected DsNvBufSurfaceGstBuffer or int (raw GstBuffer* pointer)",
        )
    })?;
    if raw == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "GstBuffer pointer is null (0); pass a valid DsNvBufSurfaceGstBuffer \
             or a non-zero raw pointer",
        ));
    }
    unsafe {
        let writable = gst::ffi::gst_mini_object_is_writable(raw as *mut gst::ffi::GstMiniObject);
        if writable == glib::ffi::GFALSE {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "GstBuffer is not writable (refcount > 1). When passing a raw \
                 pointer, ensure the buffer is exclusively owned. Prefer passing \
                 a DsNvBufSurfaceGstBuffer object instead \u{2014} it handles \
                 copy-on-write automatically.",
            ));
        }
        let buf_ref = gst::BufferRef::from_mut_ptr(raw as *mut gst::ffi::GstBuffer);
        f(buf_ref)
    }
}

/// Extract a raw ``GstBuffer*`` pointer from either a ``DsNvBufSurfaceGstBuffer`` guard
/// or a plain ``int``.
pub(crate) fn extract_buf_ptr(ob: &Bound<'_, PyAny>) -> PyResult<usize> {
    if let Ok(guard) = ob.extract::<PyRef<'_, PyDsNvBufSurfaceGstBuffer>>() {
        return guard.ptr_usize();
    }
    let raw = ob.extract::<usize>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "expected DsNvBufSurfaceGstBuffer or int (raw pointer)",
        )
    })?;
    if raw == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("buf_ptr is null"));
    }
    Ok(raw)
}

/// Extract a [`SharedBuffer`] from a Python buffer argument.
///
/// When `buf` is a [`PyDsNvBufSurfaceGstBuffer`] the inner shared buffer
/// is cloned (cheap Arc clone — GstBuffer refcount stays at 1).
/// For raw `usize` pointers a new `SharedBuffer` is created
/// from the transferred-ownership buffer.
pub(crate) fn extract_shared_buffer(buf: &Bound<'_, PyAny>) -> PyResult<SharedBuffer> {
    if let Ok(guard) = buf.extract::<PyRef<'_, PyDsNvBufSurfaceGstBuffer>>() {
        return Ok(guard.shared()?.clone());
    }
    let gst_buf = extract_gst_buffer(buf)?;
    Ok(SharedBuffer::from(gst_buf))
}

/// Extract a `gst::Buffer` from a Python buffer argument with correct
/// refcount handling.
///
/// When `buf` is a [`PyDsNvBufSurfaceGstBuffer`] the inner buffer pointer
/// is borrowed (`from_glib_none` — refcount incremented), because the
/// Python object retains ownership. For raw `usize` pointers the buffer is
/// assumed to be transferred (`from_glib_full` — no extra refcount).
///
/// GStreamer must already be initialised before calling this function.
pub(crate) fn extract_gst_buffer(buf: &Bound<'_, PyAny>) -> PyResult<gst::Buffer> {
    let is_guard = buf
        .extract::<PyRef<'_, PyDsNvBufSurfaceGstBuffer>>()
        .is_ok();
    let buf_ptr = extract_buf_ptr(buf)?;
    let gst_buf = unsafe {
        if is_guard {
            from_glib_none(buf_ptr as *const gst::ffi::GstBuffer)
        } else {
            gst::Buffer::from_glib_full(buf_ptr as *mut gst::ffi::GstBuffer)
        }
    };
    Ok(gst_buf)
}
