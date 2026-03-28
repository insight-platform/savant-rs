//! `PySharedBuffer` — safe Python wrapper for `SharedBuffer`, plus utility
//! functions for extracting buffer pointers and shared references from Python
//! arguments.

use super::enums::{from_rust_id_kind, to_rust_id_kind, PySavantIdMetaKind};
use deepstream_buffers::SharedBuffer;
use glib::translate::from_glib_none;
use gstreamer as gst;
use pyo3::prelude::*;

// ─── PySharedBuffer ─────────────────────────────────────────────────────

/// Safe Python wrapper for a `SharedBuffer`.
///
/// Uses the `Option<T>` pattern to emulate Rust move semantics in Python.
/// After a consuming Rust method (e.g. `nvinfer.submit`) calls
/// [`take_inner`](Self::take_inner), the wrapper becomes empty and all
/// subsequent property access raises `RuntimeError`.
///
/// Python code cannot construct, clone, or deconstruct this type.
#[pyclass(name = "SharedBuffer", module = "savant_rs.deepstream")]
pub struct PySharedBuffer(Option<SharedBuffer>);

const CONSUMED_MSG: &str = "SharedBuffer has been consumed";

impl PySharedBuffer {
    /// Wrap a Rust `SharedBuffer` for Python.  Only callable from Rust.
    pub fn from_rust(shared: SharedBuffer) -> Self {
        Self(Some(shared))
    }

    /// Borrow the inner `SharedBuffer`.
    ///
    /// Callers that need a [`SurfaceView`] should clone this (cheap Arc
    /// clone) and pass it to `SurfaceView::from_buffer`.
    pub(crate) fn shared(&self) -> PyResult<&SharedBuffer> {
        self.0
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(CONSUMED_MSG))
    }

    /// Consume the inner `SharedBuffer`, leaving the wrapper empty.
    ///
    /// Logs `debug!` on success and `warn!` on double-consume.
    pub(crate) fn take_inner(&mut self) -> PyResult<SharedBuffer> {
        let self_ptr = self as *const Self as usize;

        let current = self.0.as_ref().ok_or_else(|| {
            log::warn!("SharedBuffer@{self_ptr:#x}: attempt to consume already-consumed buffer");
            pyo3::exceptions::PyRuntimeError::new_err(CONSUMED_MSG)
        })?;

        let gst_ptr = {
            let guard = current.lock();
            guard.as_ref().as_ptr() as usize
        };
        let pts = current.pts_ns();

        let shared = self.0.take().unwrap();
        log::debug!("SharedBuffer@{self_ptr:#x} consumed: gst_ptr={gst_ptr:#x}, pts={pts:?}");
        Ok(shared)
    }

    /// Restore a previously taken `SharedBuffer` (e.g. on failed `into_buffer`).
    #[allow(dead_code)]
    pub(crate) fn restore(&mut self, shared: SharedBuffer) {
        self.0 = Some(shared);
    }
}

#[pymethods]
impl PySharedBuffer {
    /// Number of strong `Arc` references to the underlying buffer.
    #[getter]
    fn strong_count(&self) -> PyResult<usize> {
        Ok(self.shared()?.strong_count())
    }

    /// Buffer PTS in nanoseconds, or ``None`` if unset.
    #[getter]
    fn pts_ns(&self) -> PyResult<Option<u64>> {
        Ok(self.shared()?.pts_ns())
    }

    /// Set the buffer PTS in nanoseconds.
    #[setter]
    fn set_pts_ns(&self, pts_ns: u64) -> PyResult<()> {
        self.shared()?.set_pts_ns(pts_ns);
        Ok(())
    }

    /// Buffer duration in nanoseconds, or ``None`` if unset.
    #[getter]
    fn duration_ns(&self) -> PyResult<Option<u64>> {
        Ok(self.shared()?.duration_ns())
    }

    /// Set the buffer duration in nanoseconds.
    #[setter]
    fn set_duration_ns(&self, duration_ns: u64) -> PyResult<()> {
        self.shared()?.set_duration_ns(duration_ns);
        Ok(())
    }

    /// Read ``SavantIdMeta`` from the buffer.
    ///
    /// Returns:
    ///     list[tuple[SavantIdMetaKind, int]]: Meta entries,
    ///         e.g. ``[(SavantIdMetaKind.FRAME, 42)]``.
    fn savant_ids(&self) -> PyResult<Vec<(PySavantIdMetaKind, i64)>> {
        let ids = self.shared()?.savant_ids();
        Ok(ids.iter().map(from_rust_id_kind).collect())
    }

    /// Replace ``SavantIdMeta`` on the buffer.
    ///
    /// Args:
    ///     ids (list[tuple[SavantIdMetaKind, int]]): Meta entries to set.
    fn set_savant_ids(&self, ids: Vec<(PySavantIdMetaKind, i64)>) -> PyResult<()> {
        let kinds = ids
            .into_iter()
            .map(|(kind, id)| to_rust_id_kind(kind, id))
            .collect();
        self.shared()?.set_savant_ids(kinds);
        Ok(())
    }

    /// ``True`` if the buffer has been consumed (inner is ``None``).
    #[getter]
    fn is_consumed(&self) -> bool {
        self.0.is_none()
    }

    fn __bool__(&self) -> bool {
        self.0.is_some()
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            Some(shared) => format!("SharedBuffer(strong_count={})", shared.strong_count()),
            None => "SharedBuffer(<consumed>)".to_string(),
        }
    }
}

// ─── Buffer extraction utilities ─────────────────────────────────────────

/// Obtain a `&mut gst::BufferRef` from a Python buffer argument.
///
/// When `buf` is a [`PySharedBuffer`] the inner `gst::Buffer` is
/// accessed via the `SharedBuffer` mutex lock.  Because the
/// GstBuffer GLib refcount is always 1, `make_mut()` succeeds in-place
/// without COW.  For raw integer pointers writability is checked explicitly.
pub(crate) fn with_mut_buffer_ref<F, R>(buf: &Bound<'_, PyAny>, f: F) -> PyResult<R>
where
    F: FnOnce(&mut gst::BufferRef) -> PyResult<R>,
{
    if let Ok(guard) = buf.extract::<PyRef<'_, PySharedBuffer>>() {
        let shared = guard.shared()?;
        let mut lock = shared.lock();
        let buf_ref = lock.make_mut();
        return f(buf_ref);
    }

    let raw = buf.extract::<usize>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "expected SharedBuffer or int (raw GstBuffer* pointer)",
        )
    })?;
    if raw == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "GstBuffer pointer is null (0); pass a valid SharedBuffer \
             or a non-zero raw pointer",
        ));
    }
    unsafe {
        let writable = gst::ffi::gst_mini_object_is_writable(raw as *mut gst::ffi::GstMiniObject);
        if writable == glib::ffi::GFALSE {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "GstBuffer is not writable (refcount > 1). When passing a raw \
                 pointer, ensure the buffer is exclusively owned. Prefer passing \
                 a SharedBuffer object instead \u{2014} it handles \
                 copy-on-write automatically.",
            ));
        }
        let buf_ref = gst::BufferRef::from_mut_ptr(raw as *mut gst::ffi::GstBuffer);
        f(buf_ref)
    }
}

/// Extract a raw ``GstBuffer*`` pointer from either a ``SharedBuffer`` or
/// a plain ``int``.
pub(crate) fn extract_buf_ptr(ob: &Bound<'_, PyAny>) -> PyResult<usize> {
    if let Ok(guard) = ob.extract::<PyRef<'_, PySharedBuffer>>() {
        let shared = guard.shared()?;
        let lock = shared.lock();
        return Ok(lock.as_ref().as_ptr() as usize);
    }
    let raw = ob.extract::<usize>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err("expected SharedBuffer or int (raw pointer)")
    })?;
    if raw == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("buf_ptr is null"));
    }
    Ok(raw)
}

/// Extract a [`SharedBuffer`] from a Python buffer argument.
///
/// When `buf` is a [`PySharedBuffer`] the inner shared buffer
/// is cloned (cheap Arc clone — GstBuffer refcount stays at 1).
/// For raw `usize` pointers a new `SharedBuffer` is created
/// from the transferred-ownership buffer.
pub(crate) fn extract_shared_buffer(buf: &Bound<'_, PyAny>) -> PyResult<SharedBuffer> {
    if let Ok(guard) = buf.extract::<PyRef<'_, PySharedBuffer>>() {
        return Ok(guard.shared()?.clone());
    }
    let gst_buf = extract_gst_buffer(buf)?;
    Ok(SharedBuffer::from(gst_buf))
}

/// Extract a `gst::Buffer` from a Python buffer argument with correct
/// refcount handling.
///
/// When `buf` is a [`PySharedBuffer`] the inner buffer pointer
/// is borrowed (`from_glib_none` — refcount incremented), because the
/// Python object retains ownership. For raw `usize` pointers the buffer is
/// assumed to be transferred (`from_glib_full` — no extra refcount).
///
/// GStreamer must already be initialised before calling this function.
pub(crate) fn extract_gst_buffer(buf: &Bound<'_, PyAny>) -> PyResult<gst::Buffer> {
    let is_shared = buf.extract::<PyRef<'_, PySharedBuffer>>().is_ok();
    let buf_ptr = extract_buf_ptr(buf)?;
    let gst_buf = unsafe {
        if is_shared {
            from_glib_none(buf_ptr as *const gst::ffi::GstBuffer)
        } else {
            gst::Buffer::from_glib_full(buf_ptr as *mut gst::ffi::GstBuffer)
        }
    };
    Ok(gst_buf)
}
