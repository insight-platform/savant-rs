//! Shared mutable GStreamer buffer wrapper.
//!
//! [`SharedMutableGstBuffer`] is a newtype around `Arc<Mutex<gst::Buffer>>`
//! that serves as the shared currency for passing NvBufSurface-backed buffers
//! between [`SurfaceView`], Picasso, and the encoder without ownership transfer.
//!
//! [`SurfaceView`]: crate::SurfaceView

use crate::{SavantIdMeta, SavantIdMetaKind};
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
/// # use deepstream_nvbufsurface::SharedMutableGstBuffer;
/// gst::init().unwrap();
/// let buf = gst::Buffer::new();
/// let shared = SharedMutableGstBuffer::from(buf);
/// ```
///
/// # Cloning
///
/// `Clone` increments the `Arc` reference count (cheap, no data copy).
/// Multiple clones share the same underlying `gst::Buffer`.
///
/// [`SurfaceView`]: crate::SurfaceView
#[derive(Clone)]
pub struct SharedMutableGstBuffer(Arc<Mutex<gst::Buffer>>);

// SAFETY: Arc<Mutex<T>> is Send + Sync when T: Send.
// gst::Buffer is Send (it wraps a refcounted GstMiniObject).
unsafe impl Send for SharedMutableGstBuffer {}
unsafe impl Sync for SharedMutableGstBuffer {}

impl SharedMutableGstBuffer {
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

    /// Read [`SavantIdMeta`] from the buffer and return it as `(kind, id)` pairs.
    ///
    /// Returns `("frame", id)` for [`SavantIdMetaKind::Frame`] and
    /// `("batch", id)` for [`SavantIdMetaKind::Batch`].
    pub fn savant_ids(&self) -> Vec<(String, i64)> {
        let guard = self.lock();
        match guard.meta::<SavantIdMeta>() {
            Some(meta) => meta
                .ids()
                .iter()
                .map(|k| match k {
                    SavantIdMetaKind::Frame(id) => ("frame".to_string(), *id),
                    SavantIdMetaKind::Batch(id) => ("batch".to_string(), *id),
                })
                .collect(),
            None => vec![],
        }
    }
}

impl From<gst::Buffer> for SharedMutableGstBuffer {
    fn from(buf: gst::Buffer) -> Self {
        Self(Arc::new(Mutex::new(buf)))
    }
}

impl fmt::Debug for SharedMutableGstBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SharedMutableGstBuffer")
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
        let shared = SharedMutableGstBuffer::from(buf);
        assert_eq!(shared.strong_count(), 1);
        let _buf = shared.into_buffer().expect("sole owner should succeed");
    }

    #[test]
    fn test_into_buffer_fails_with_clones() {
        gst::init().unwrap();
        let buf = gst::Buffer::new();
        let shared = SharedMutableGstBuffer::from(buf);
        let _clone = shared.clone();
        assert_eq!(shared.strong_count(), 2);
        let err = shared.into_buffer().unwrap_err();
        assert_eq!(err.strong_count(), 2);
    }

    #[test]
    fn test_into_buffer_succeeds_after_clone_dropped() {
        gst::init().unwrap();
        let buf = gst::Buffer::new();
        let shared = SharedMutableGstBuffer::from(buf);
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
        let shared = SharedMutableGstBuffer::from(buf);
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
        let shared = SharedMutableGstBuffer::from(gst::Buffer::new());
        let clone = shared.clone();
        assert_eq!(shared.strong_count(), 2);
        assert_eq!(clone.strong_count(), 2);
        drop(clone);
        assert_eq!(shared.strong_count(), 1);
    }

    #[test]
    fn test_debug_format() {
        gst::init().unwrap();
        let shared = SharedMutableGstBuffer::from(gst::Buffer::new());
        let dbg = format!("{:?}", shared);
        assert!(dbg.contains("SharedMutableGstBuffer"));
        assert!(dbg.contains("strong_count"));
    }

    #[test]
    fn test_pts_ns_roundtrip() {
        gst::init().unwrap();
        let shared = SharedMutableGstBuffer::from(gst::Buffer::new());
        assert_eq!(shared.pts_ns(), None);
        shared.set_pts_ns(42_000);
        assert_eq!(shared.pts_ns(), Some(42_000));
    }

    #[test]
    fn test_duration_ns_roundtrip() {
        gst::init().unwrap();
        let shared = SharedMutableGstBuffer::from(gst::Buffer::new());
        assert_eq!(shared.duration_ns(), None);
        shared.set_duration_ns(33_333_333);
        assert_eq!(shared.duration_ns(), Some(33_333_333));
    }
}
