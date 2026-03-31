//! Shared mutable GStreamer buffer wrapper.
//!
//! [`SharedBuffer`] is a newtype around `Arc<Mutex<gst::Buffer>>`
//! that serves as the shared currency for passing NvBufSurface-backed buffers
//! between [`SurfaceView`], Picasso, and the encoder without ownership transfer.
//!
//! [`SurfaceView`]: crate::SurfaceView

use crate::{Rect, SavantIdMeta, SavantIdMetaKind, TransformConfig};
use gstreamer as gst;
use parking_lot::{Mutex, MutexGuard};
use std::fmt;
use std::sync::Arc;

/// Shared mutable reference to a GStreamer buffer.
///
/// Wraps `Arc<Mutex<gst::Buffer>>` so multiple [`SurfaceView`]s (for
/// different batch slots) can reference the same underlying buffer, and
/// the buffer can be passed to downstream consumers (encoder, NvInfer)
/// without extracting it from a view first.
///
/// # Construction
///
/// ```rust,no_run
/// # use gstreamer as gst;
/// # use deepstream_buffers::SharedBuffer;
/// gst::init().unwrap();
/// let buf = gst::Buffer::new();
/// let shared = SharedBuffer::from(buf);
/// ```
///
/// # Cloning
///
/// `Clone` increments the `Arc` reference count (cheap, no data copy).
/// Multiple clones share the same underlying `gst::Buffer`.
///
/// [`SurfaceView`]: crate::SurfaceView
#[derive(Clone)]
pub struct SharedBuffer(Arc<Mutex<gst::Buffer>>);

// SAFETY: Arc<Mutex<T>> is Send + Sync when T: Send.
// gst::Buffer is Send (it wraps a refcounted GstMiniObject).
unsafe impl Send for SharedBuffer {}
unsafe impl Sync for SharedBuffer {}

impl SharedBuffer {
    /// Lock the inner buffer for reading or writing.
    ///
    /// The returned guard auto-derefs to `&gst::Buffer` (via `Deref`) and
    /// `&mut gst::Buffer` (via `DerefMut`), so callers can call
    /// `guard.as_ref()`, `guard.make_mut()`, etc. directly.
    pub fn lock(&self) -> MutexGuard<'_, gst::Buffer> {
        self.0.lock()
    }

    /// Consume this handle and extract the inner `gst::Buffer`.
    ///
    /// Succeeds only when this is the **sole** strong reference (i.e. all
    /// sibling [`SurfaceView`]s and clones have been dropped).  Returns
    /// `Err(self)` otherwise — the caller must drop outstanding references
    /// before retrying.
    ///
    /// [`SurfaceView`]: crate::SurfaceView
    pub fn into_buffer(self) -> Result<gst::Buffer, Self> {
        match Arc::try_unwrap(self.0) {
            Ok(mutex) => Ok(mutex.into_inner()),
            Err(arc) => Err(Self(arc)),
        }
    }

    /// Number of strong references to the underlying buffer.
    ///
    /// Useful for diagnostics; 1 means this is the sole owner.
    pub fn strong_count(&self) -> usize {
        Arc::strong_count(&self.0)
    }

    /// Return the buffer PTS in nanoseconds, or `None` if unset.
    pub fn pts_ns(&self) -> Option<u64> {
        let guard = self.lock();
        guard.as_ref().pts().map(|t| t.nseconds())
    }

    /// Set the buffer PTS in nanoseconds.
    pub fn set_pts_ns(&self, pts_ns: u64) {
        let mut guard = self.lock();
        guard
            .make_mut()
            .set_pts(gst::ClockTime::from_nseconds(pts_ns));
    }

    /// Return the buffer duration in nanoseconds, or `None` if unset.
    pub fn duration_ns(&self) -> Option<u64> {
        let guard = self.lock();
        guard.as_ref().duration().map(|t| t.nseconds())
    }

    /// Set the buffer duration in nanoseconds.
    pub fn set_duration_ns(&self, duration_ns: u64) {
        let mut guard = self.lock();
        guard
            .make_mut()
            .set_duration(gst::ClockTime::from_nseconds(duration_ns));
    }

    /// Read [`SavantIdMeta`] from the buffer.
    ///
    /// Returns the stored IDs, or an empty vec if no meta is attached.
    pub fn savant_ids(&self) -> Vec<SavantIdMetaKind> {
        let guard = self.lock();
        match guard.meta::<SavantIdMeta>() {
            Some(meta) => meta.ids().to_vec(),
            None => vec![],
        }
    }

    /// Create a temporary [`SurfaceView`] and pass it to the closure.
    ///
    /// The view is created, handed to `f`, and **automatically dropped**
    /// when the closure returns — no manual `drop(view)` needed.  This
    /// keeps the `Arc` refcount elevated only for the duration of `f`.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use deepstream_buffers::{BufferGenerator, NvBufSurfaceMemType, VideoFormat};
    /// # gstreamer::init().unwrap();
    /// # let gen = BufferGenerator::new(
    /// #     VideoFormat::RGBA, 640, 480, 30, 1, 0, NvBufSurfaceMemType::Default,
    /// # ).unwrap();
    /// let shared = gen.acquire(None).unwrap();
    /// shared.with_view(0, |view| {
    ///     view.memset(0)
    /// }).unwrap();
    /// // shared is sole owner again here
    /// ```
    ///
    /// [`SurfaceView`]: crate::SurfaceView
    pub fn with_view<F, R>(&self, slot: u32, f: F) -> Result<R, crate::NvBufSurfaceError>
    where
        F: FnOnce(&crate::SurfaceView) -> Result<R, crate::NvBufSurfaceError>,
    {
        let view = crate::SurfaceView::from_buffer(self, slot)?;
        f(&view)
    }

    /// Transform one source slot into one destination slot using
    /// `NvBufSurfTransform`, without requiring EGL-CUDA mapping.
    ///
    /// This path is intended for decoder outputs that can be block-linear
    /// on Jetson, where [`crate::SurfaceView::from_buffer`] may fail due to
    /// pitch-only EGL frame constraints.
    pub fn transform_into(
        &self,
        src_slot: u32,
        dst: &SharedBuffer,
        dst_slot: u32,
        config: &TransformConfig,
        src_rect: Option<&Rect>,
    ) -> Result<(), crate::NvBufSurfaceError> {
        if Arc::ptr_eq(&self.0, &dst.0) {
            return Err(crate::NvBufSurfaceError::InvalidInput(
                "in-place SharedBuffer transform is not supported".to_string(),
            ));
        }

        let src_guard = self.lock();
        let dst_guard = dst.lock();

        let src_surf_ptr = unsafe {
            crate::transform::extract_nvbufsurface(src_guard.as_ref())
                .map_err(|e| crate::NvBufSurfaceError::BufferCopyFailed(e.to_string()))?
        };
        let dst_surf_ptr = unsafe {
            crate::transform::extract_nvbufsurface(dst_guard.as_ref())
                .map_err(|e| crate::NvBufSurfaceError::BufferCopyFailed(e.to_string()))?
        };

        let mut eff_config = config.clone();
        eff_config.cuda_stream = config.cuda_stream.clone();

        let (mut src_view, mut dst_view) = unsafe {
            let mut src = *src_surf_ptr;
            if src_slot >= src.numFilled {
                return Err(crate::NvBufSurfaceError::SlotOutOfBounds {
                    index: src_slot,
                    max: src.numFilled,
                });
            }
            src.surfaceList = src.surfaceList.add(src_slot as usize);
            src.batchSize = 1;
            src.numFilled = 1;

            let mut dst_s = *dst_surf_ptr;
            if dst_slot >= dst_s.numFilled {
                return Err(crate::NvBufSurfaceError::SlotOutOfBounds {
                    index: dst_slot,
                    max: dst_s.numFilled,
                });
            }
            dst_s.surfaceList = dst_s.surfaceList.add(dst_slot as usize);
            dst_s.batchSize = 1;
            dst_s.numFilled = 1;
            (src, dst_s)
        };

        unsafe {
            crate::transform::do_transform(
                &mut src_view as *mut crate::ffi::NvBufSurface,
                &mut dst_view as *mut crate::ffi::NvBufSurface,
                &eff_config,
                src_rect,
            )
        }
        .map_err(|e| crate::NvBufSurfaceError::BufferCopyFailed(e.to_string()))
    }

    /// Replace [`SavantIdMeta`] on the buffer.
    ///
    /// Removes any previously attached meta before adding the new IDs.
    pub fn set_savant_ids(&self, ids: Vec<SavantIdMetaKind>) {
        let mut guard = self.lock();
        let buf = guard.make_mut();
        if let Some(old) = buf.meta_mut::<SavantIdMeta>() {
            old.remove().ok();
        }
        SavantIdMeta::replace(buf, ids);
    }
}

impl From<gst::Buffer> for SharedBuffer {
    fn from(buf: gst::Buffer) -> Self {
        Self(Arc::new(Mutex::new(buf)))
    }
}

impl fmt::Debug for SharedBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SharedBuffer")
            .field("strong_count", &Arc::strong_count(&self.0))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_buffer_and_into_buffer() {
        gst::init().unwrap();
        let buf = gst::Buffer::new();
        let shared = SharedBuffer::from(buf);
        assert_eq!(shared.strong_count(), 1);
        let _buf = shared.into_buffer().expect("sole owner should succeed");
    }

    #[test]
    fn test_into_buffer_fails_with_clones() {
        gst::init().unwrap();
        let buf = gst::Buffer::new();
        let shared = SharedBuffer::from(buf);
        let _clone = shared.clone();
        assert_eq!(shared.strong_count(), 2);
        let err = shared.into_buffer().unwrap_err();
        assert_eq!(err.strong_count(), 2);
    }

    #[test]
    fn test_into_buffer_succeeds_after_clone_dropped() {
        gst::init().unwrap();
        let buf = gst::Buffer::new();
        let shared = SharedBuffer::from(buf);
        let clone = shared.clone();
        drop(clone);
        let _buf = shared
            .into_buffer()
            .expect("should succeed after clone dropped");
    }

    #[test]
    fn test_lock_read_write() {
        gst::init().unwrap();
        let buf = gst::Buffer::new();
        let shared = SharedBuffer::from(buf);
        {
            let mut guard = shared.lock();
            let buf_ref = guard.make_mut();
            buf_ref.set_pts(gst::ClockTime::from_nseconds(42_000));
        }
        {
            let guard = shared.lock();
            assert_eq!(
                guard.as_ref().pts(),
                Some(gst::ClockTime::from_nseconds(42_000))
            );
        }
    }

    #[test]
    fn test_clone_is_cheap() {
        gst::init().unwrap();
        let shared = SharedBuffer::from(gst::Buffer::new());
        let clone = shared.clone();
        assert_eq!(shared.strong_count(), 2);
        assert_eq!(clone.strong_count(), 2);
        drop(clone);
        assert_eq!(shared.strong_count(), 1);
    }

    #[test]
    fn test_debug_format() {
        gst::init().unwrap();
        let shared = SharedBuffer::from(gst::Buffer::new());
        let dbg = format!("{:?}", shared);
        assert!(dbg.contains("SharedBuffer"));
        assert!(dbg.contains("strong_count"));
    }

    #[test]
    fn test_pts_ns_roundtrip() {
        gst::init().unwrap();
        let shared = SharedBuffer::from(gst::Buffer::new());
        assert_eq!(shared.pts_ns(), None);
        shared.set_pts_ns(42_000);
        assert_eq!(shared.pts_ns(), Some(42_000));
    }

    #[test]
    fn test_duration_ns_roundtrip() {
        gst::init().unwrap();
        let shared = SharedBuffer::from(gst::Buffer::new());
        assert_eq!(shared.duration_ns(), None);
        shared.set_duration_ns(33_333_333);
        assert_eq!(shared.duration_ns(), Some(33_333_333));
    }

    #[test]
    fn test_savant_ids_empty_by_default() {
        gst::init().unwrap();
        let shared = SharedBuffer::from(gst::Buffer::new());
        assert!(shared.savant_ids().is_empty());
    }

    #[test]
    fn test_savant_ids_roundtrip() {
        gst::init().unwrap();
        let shared = SharedBuffer::from(gst::Buffer::new());
        let ids = vec![SavantIdMetaKind::Frame(1), SavantIdMetaKind::Batch(2)];
        shared.set_savant_ids(ids.clone());
        assert_eq!(shared.savant_ids(), ids);
    }

    #[test]
    fn test_savant_ids_replace() {
        gst::init().unwrap();
        let shared = SharedBuffer::from(gst::Buffer::new());
        shared.set_savant_ids(vec![SavantIdMetaKind::Frame(1)]);
        shared.set_savant_ids(vec![SavantIdMetaKind::Frame(99)]);
        assert_eq!(shared.savant_ids(), vec![SavantIdMetaKind::Frame(99)]);
    }

    /// `with_view` requires a real NvBufSurface-backed buffer (GPU), so the
    /// ownership-restoring contract is verified via integration tests.  Here
    /// we only check that view-creation errors propagate correctly.
    #[test]
    fn test_with_view_propagates_view_error() {
        gst::init().unwrap();
        let shared = SharedBuffer::from(gst::Buffer::new());
        let result: Result<(), _> = shared.with_view(0, |_view| Ok(()));
        assert!(result.is_err(), "empty buffer should fail from_buffer");
        assert_eq!(shared.strong_count(), 1, "no lingering Arc clones on error");
    }
}
