use crate::{DeepStreamError, FrameMeta, Result, UserMeta};
use deepstream_sys::{
    gst_buffer_get_nvds_batch_meta, nvds_clear_obj_meta_list, GstBuffer, NvDsBatchMeta,
    NvDsFrameMeta,
};
use std::sync::{Arc, Once};

/// Force-initialise the private tag quark inside `libnvdsgst_meta.so`.
///
/// Any code path that calls `gst_buffer_get_nvds_batch_meta` must call this
/// first to avoid `gst_meta_api_type_has_tag: assertion 'tag != 0' failed`.
///
/// Safe to call many times; the underlying init runs only once process-wide.
pub fn ensure_nvds_meta_api_registered() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        // SAFETY: nvds_meta_api_get_type is safe to call once for global init.
        unsafe { deepstream_sys::nvds_meta_api_get_type() };
    });
}

/// Remove every `NvDsObjectMeta` from every frame in the batch meta attached
/// to `buf_ptr`.
///
/// Used by `MetaClearPolicy::Before` (before submitting to inference/tracker)
/// and `MetaClearPolicy::After` (when the output is dropped) to keep the
/// batch buffer free of stale upstream object metas.
///
/// No-op when `buf_ptr` is null or has no attached batch meta.
///
/// # Safety
/// `buf_ptr` must be a valid, writable `*mut GstBuffer` with a unique
/// reference (i.e. writable per GStreamer's copy-on-write rules). Concurrent
/// access to the batch meta linked lists is not safe.
pub unsafe fn clear_all_frame_objects(buf_ptr: *mut GstBuffer) {
    ensure_nvds_meta_api_registered();
    if buf_ptr.is_null() {
        return;
    }
    let batch_meta = gst_buffer_get_nvds_batch_meta(buf_ptr);
    if batch_meta.is_null() {
        return;
    }
    clear_frames_objects(batch_meta);
}

/// Iterate the frame meta list of `batch_meta` and clear every
/// `NvDsObjectMeta` via `nvds_clear_obj_meta_list`.
fn clear_frames_objects(batch_meta: *mut NvDsBatchMeta) {
    let mut frame_list = unsafe { (*batch_meta).frame_meta_list };
    while !frame_list.is_null() {
        let frame_ptr = unsafe { (*frame_list).data as *mut NvDsFrameMeta };
        if !frame_ptr.is_null() {
            let obj_list = unsafe { (*frame_ptr).obj_meta_list };
            if !obj_list.is_null() {
                unsafe { nvds_clear_obj_meta_list(frame_ptr, obj_list) };
            }
        }
        frame_list = unsafe { (*frame_list).next };
    }
}

/// RAII guard for the DeepStream batch metadata lock.
///
/// Acquires `nvds_acquire_meta_lock` on construction, releases via
/// `nvds_release_meta_lock` on drop. The lock is held for the lifetime of
/// every `BatchMeta` (and its clones) to guarantee safe traversal of the
/// metadata linked lists.
struct BatchMetaLock(*mut NvDsBatchMeta);

// SAFETY: The raw pointer is to DeepStream-managed memory allocated on the
// heap. Sending/sharing the guard across threads is safe because:
// 1. The pointer itself is just a usize-sized value.
// 2. We never dereference it except through the lock/unlock FFI calls.
// 3. The DS lock serialises all concurrent access to the batch metadata.
unsafe impl Send for BatchMetaLock {}
// SAFETY: BatchMetaLock contains only a raw pointer and only interacts with
// it through the acquire/release FFI calls. No &self methods read or write
// through the pointer, so sharing references across threads is safe.
unsafe impl Sync for BatchMetaLock {}

impl BatchMetaLock {
    fn new(raw: *mut NvDsBatchMeta) -> Self {
        // SAFETY: raw is non-null (checked by caller in BatchMeta::from_gst_buffer).
        unsafe {
            deepstream_sys::nvds_acquire_meta_lock(raw);
        }
        Self(raw)
    }
}

impl Drop for BatchMetaLock {
    fn drop(&mut self) {
        // SAFETY: We acquired the lock in `new`; releasing it is always valid.
        unsafe {
            deepstream_sys::nvds_release_meta_lock(self.0);
        }
    }
}

/// Safe wrapper for DeepStream batch metadata.
///
/// Cloning is shallow: all clones share the same underlying batch and keep
/// the DS metadata lock alive until the last clone is dropped.
pub struct BatchMeta {
    raw: *mut NvDsBatchMeta,
    /// Shared ownership of the lock guard; kept alive via `Arc`.
    _lock: Arc<BatchMetaLock>,
}

impl BatchMeta {
    /// Obtain batch metadata from a `GstBuffer`.
    ///
    /// # Safety
    ///
    /// `buffer` must be a valid, non-null pointer to a `GstBuffer` that
    /// contains DeepStream batch metadata.
    pub unsafe fn from_gst_buffer(buffer: *mut GstBuffer) -> Result<Self> {
        ensure_nvds_meta_api_registered();
        let raw = deepstream_sys::gst_buffer_get_nvds_batch_meta(buffer);
        if raw.is_null() {
            return Err(DeepStreamError::null_pointer(
                "No batch metadata found in buffer",
            ));
        }
        Ok(Self {
            raw,
            _lock: Arc::new(BatchMetaLock::new(raw)),
        })
    }

    /// Raw pointer to the underlying C struct.
    pub fn as_raw(&self) -> *mut NvDsBatchMeta {
        self.raw
    }

    /// Number of frames in the batch.
    pub fn num_frames(&self) -> u32 {
        unsafe { (*self.raw).num_frames_in_batch }
    }

    /// Maximum frames that can fit in the batch.
    pub fn max_frames_in_batch(&self) -> u32 {
        unsafe { (*self.raw).max_frames_in_batch }
    }

    /// Iterate over all frame metadata in the batch.
    pub fn frames(&self) -> Vec<FrameMeta> {
        let mut frames = Vec::new();
        let mut current = unsafe { (*self.raw).frame_meta_list };
        while !current.is_null() {
            let data = unsafe { (*current).data };
            if !data.is_null() {
                if let Ok(frame) =
                    unsafe { FrameMeta::from_raw(data as *mut deepstream_sys::NvDsFrameMeta, self) }
                {
                    frames.push(frame);
                }
            }
            current = unsafe { (*current).next };
        }
        frames
    }

    /// Whether the batch contains zero frames.
    pub fn is_empty(&self) -> bool {
        self.num_frames() == 0
    }

    /// Batch-level user metadata (e.g. tracker shadow / terminated lists).
    pub fn batch_user_meta(&self) -> Vec<UserMeta> {
        let mut list = Vec::new();
        let mut current = unsafe { (*self.raw).batch_user_meta_list };
        while !current.is_null() {
            let data = unsafe { (*current).data };
            if !data.is_null() {
                if let Ok(um) =
                    unsafe { UserMeta::from_raw(data as *mut deepstream_sys::NvDsUserMeta, self) }
                {
                    list.push(um);
                }
            }
            current = unsafe { (*current).next };
        }
        list
    }
}

impl Clone for BatchMeta {
    fn clone(&self) -> Self {
        Self {
            raw: self.raw,
            _lock: self._lock.clone(),
        }
    }
}

impl std::fmt::Debug for BatchMeta {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchMeta")
            .field("num_frames", &self.num_frames())
            .field("max_frames_in_batch", &self.max_frames_in_batch())
            .finish()
    }
}
