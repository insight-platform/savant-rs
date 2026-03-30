//! Safe RAII wrapper around a CUDA stream handle.
//!
//! [`CudaStream`] manages the lifecycle of a CUDA stream, providing safe
//! creation, synchronization, and automatic destruction on drop.
//!
//! # Ownership model
//!
//! - **Owned** streams (from [`CudaStream::new_non_blocking`]) are destroyed
//!   on [`Drop`].
//! - **Borrowed** streams (from [`Default`] or [`CudaStream::from_raw`]) do
//!   **not** destroy the underlying handle.
//! - [`Clone`] always produces a non-owning copy.

use crate::ffi;
use crate::NvBufSurfaceError;

/// Safe RAII wrapper around a CUDA stream handle.
///
/// The default stream (constructed via [`Default::default`]) is the CUDA
/// legacy stream (null pointer) and is non-owning.
pub struct CudaStream {
    raw: *mut std::ffi::c_void,
    owned: bool,
}

// SAFETY: CUDA streams are thread-safe — work can be enqueued from any
// thread.  The raw pointer is an opaque handle, not a Rust-side reference.
unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

impl CudaStream {
    /// Create a non-blocking CUDA stream (`cudaStreamNonBlocking`).
    ///
    /// The returned handle is **owned** and will be destroyed on [`Drop`].
    /// Non-blocking streams do not implicitly synchronize with the CUDA
    /// legacy default stream (stream 0), eliminating the global GPU
    /// serialization barrier.
    ///
    /// # Errors
    ///
    /// Returns [`NvBufSurfaceError::CudaInitFailed`] if stream creation fails.
    pub fn new_non_blocking() -> Result<Self, NvBufSurfaceError> {
        let mut raw: *mut std::ffi::c_void = std::ptr::null_mut();
        // 0x01 = cudaStreamNonBlocking
        let ret = unsafe { ffi::cudaStreamCreateWithFlags(&mut raw, 0x01) };
        if ret != 0 {
            return Err(NvBufSurfaceError::CudaInitFailed(ret));
        }
        log::debug!("Created non-blocking CUDA stream {:?}", raw);
        Ok(Self { raw, owned: true })
    }

    /// Wrap an existing raw CUDA stream pointer without taking ownership.
    ///
    /// The resulting handle will **not** destroy the stream on drop.
    ///
    /// # Safety
    ///
    /// `ptr` must be a valid CUDA stream handle or null.
    pub unsafe fn from_raw(ptr: *mut std::ffi::c_void) -> Self {
        Self {
            raw: ptr,
            owned: false,
        }
    }

    /// Returns the raw CUDA stream pointer.
    pub fn as_raw(&self) -> *mut std::ffi::c_void {
        self.raw
    }

    /// Returns `true` if this is the CUDA default (legacy) stream (null pointer).
    pub fn is_default(&self) -> bool {
        self.raw.is_null()
    }

    /// Returns `true` if this handle owns the underlying CUDA stream.
    pub fn is_owned(&self) -> bool {
        self.owned
    }

    /// Block until all previously enqueued work on this stream completes.
    ///
    /// # Errors
    ///
    /// Returns [`NvBufSurfaceError::CudaInitFailed`] if synchronization fails.
    pub fn synchronize(&self) -> Result<(), NvBufSurfaceError> {
        let err = unsafe { ffi::cudaStreamSynchronize(self.raw) };
        if err != 0 {
            return Err(NvBufSurfaceError::CudaInitFailed(err));
        }
        Ok(())
    }

    /// Block until all previously enqueued work on this stream completes,
    /// logging a warning on failure instead of returning an error.
    ///
    /// Use this variant when synchronization failure is non-fatal and the
    /// calling code cannot propagate errors.
    pub fn synchronize_or_log(&self) {
        if let Err(e) = self.synchronize() {
            log::warn!("cudaStreamSynchronize failed: {e}");
        }
    }
}

impl Default for CudaStream {
    /// The CUDA default (legacy) stream — null pointer, non-owning.
    fn default() -> Self {
        Self {
            raw: std::ptr::null_mut(),
            owned: false,
        }
    }
}

impl Clone for CudaStream {
    /// Clone always produces a **non-owning** handle.
    ///
    /// The cloned handle refers to the same underlying CUDA stream but
    /// will not destroy it on drop.
    fn clone(&self) -> Self {
        Self {
            raw: self.raw,
            owned: false,
        }
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if self.owned && !self.raw.is_null() {
            let ret = unsafe { ffi::cudaStreamDestroy(self.raw) };
            if ret != 0 {
                log::warn!("cudaStreamDestroy failed: code {ret}");
            } else {
                log::debug!("Destroyed CUDA stream {:?}", self.raw);
            }
        }
    }
}

impl std::fmt::Debug for CudaStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaStream")
            .field("raw", &self.raw)
            .field("owned", &self.owned)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_null_and_non_owning() {
        let stream = CudaStream::default();
        assert!(stream.is_default());
        assert!(!stream.is_owned());
        assert!(stream.as_raw().is_null());
    }

    #[test]
    fn clone_is_non_owning() {
        let stream = CudaStream {
            raw: 0xDEAD as *mut std::ffi::c_void,
            owned: true,
        };
        let cloned = stream.clone();
        assert_eq!(cloned.as_raw(), stream.as_raw());
        assert!(!cloned.is_owned());
        // Prevent the original from actually calling cudaStreamDestroy
        std::mem::forget(stream);
    }

    #[test]
    fn from_raw_is_non_owning() {
        let stream = unsafe { CudaStream::from_raw(0xBEEF as *mut std::ffi::c_void) };
        assert!(!stream.is_owned());
        assert!(!stream.is_default());
        assert_eq!(stream.as_raw(), 0xBEEF as *mut std::ffi::c_void);
    }

    #[test]
    fn debug_format() {
        let stream = CudaStream::default();
        let s = format!("{:?}", stream);
        assert!(s.contains("CudaStream"));
        assert!(s.contains("owned"));
    }

    #[test]
    fn synchronize_default_stream_succeeds() {
        // The null/default stream is a no-op for synchronize on CUDA-initialized systems
        // but may fail if CUDA is not available. We just verify it doesn't panic.
        let stream = CudaStream::default();
        let _ = stream.synchronize(); // May succeed or fail depending on CUDA availability
    }
}
