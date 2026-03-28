//! GStreamer AppSrc helpers for [`BufferGenerator`](super::BufferGenerator).
//!
//! Free functions for pushing buffers to an `appsrc` element and sending
//! end-of-stream signals. Separated from the core generator to keep the
//! public API surface of [`BufferGenerator`] focused on pool management.

use crate::NvBufSurfaceError;
use gstreamer as gst;

/// Push a new NVMM buffer to an AppSrc element.
///
/// Acquires a buffer from `gen`, sets PTS/duration, and pushes it to `appsrc`.
///
/// # Arguments
///
/// * `gen` - Generator to acquire the buffer from.
/// * `appsrc` - The GStreamer AppSrc element to push to.
/// * `pts_ns` - Presentation timestamp in nanoseconds.
/// * `duration_ns` - Buffer duration in nanoseconds.
/// * `id` - Optional frame identifier for SavantIdMeta attachment.
pub fn push_to_appsrc(
    gen: &super::BufferGenerator,
    appsrc: &gstreamer_app::AppSrc,
    pts_ns: u64,
    duration_ns: u64,
    id: Option<i64>,
) -> Result<(), NvBufSurfaceError> {
    let mut buffer = gen.acquire(id)?.into_buffer().map_err(|_| {
        NvBufSurfaceError::BufferAcquisitionFailed("SharedBuffer has outstanding references".into())
    })?;

    {
        let buf_ref = buffer.get_mut().unwrap();
        buf_ref.set_pts(gst::ClockTime::from_nseconds(pts_ns));
        buf_ref.set_duration(gst::ClockTime::from_nseconds(duration_ns));
    }

    appsrc.push_buffer(buffer).map_err(|e| {
        NvBufSurfaceError::BufferAcquisitionFailed(format!("AppSrc push failed: {:?}", e))
    })?;

    Ok(())
}

/// Push a new NVMM buffer to an AppSrc given as a raw pointer.
///
/// FFI-friendly variant of [`push_to_appsrc`], accepting the AppSrc as a raw
/// `GstElement*` pointer (e.g., from `hash(appsrc)` in Python).
///
/// # Safety
///
/// The caller must ensure that `appsrc_ptr` is a valid, non-null pointer
/// to a `GstAppSrc` element that remains valid for the duration of this call.
pub unsafe fn push_to_appsrc_raw(
    gen: &super::BufferGenerator,
    appsrc_ptr: usize,
    pts_ns: u64,
    duration_ns: u64,
    id: Option<i64>,
) -> Result<(), NvBufSurfaceError> {
    if appsrc_ptr == 0 {
        return Err(NvBufSurfaceError::NullPointer(
            "appsrc pointer is null".to_string(),
        ));
    }

    let mut buffer = gen.acquire(id)?.into_buffer().map_err(|_| {
        NvBufSurfaceError::BufferAcquisitionFailed("SharedBuffer has outstanding references".into())
    })?;
    {
        let buf_ref = buffer.get_mut().unwrap();
        buf_ref.set_pts(gst::ClockTime::from_nseconds(pts_ns));
        buf_ref.set_duration(gst::ClockTime::from_nseconds(duration_ns));
    }

    let buf_ptr = {
        use glib::translate::IntoGlibPtr;
        buffer.into_glib_ptr()
    };

    extern "C" {
        fn gst_app_src_push_buffer(
            appsrc: *mut gst::ffi::GstElement,
            buffer: *mut gst::ffi::GstBuffer,
        ) -> i32;
    }

    let ret = gst_app_src_push_buffer(appsrc_ptr as *mut gst::ffi::GstElement, buf_ptr);

    if ret != 0 {
        return Err(NvBufSurfaceError::BufferAcquisitionFailed(format!(
            "gst_app_src_push_buffer returned {}",
            ret
        )));
    }

    Ok(())
}

/// Send an end-of-stream signal to an AppSrc element.
pub fn send_eos(appsrc: &gstreamer_app::AppSrc) -> Result<(), NvBufSurfaceError> {
    appsrc.end_of_stream().map_err(|e| {
        NvBufSurfaceError::BufferAcquisitionFailed(format!("AppSrc EOS failed: {:?}", e))
    })?;
    Ok(())
}

/// Send an end-of-stream signal to an AppSrc given as a raw pointer.
///
/// # Safety
///
/// The caller must ensure that `appsrc_ptr` is a valid, non-null pointer
/// to a `GstAppSrc` element.
pub unsafe fn send_eos_raw(appsrc_ptr: usize) -> Result<(), NvBufSurfaceError> {
    if appsrc_ptr == 0 {
        return Err(NvBufSurfaceError::NullPointer(
            "appsrc pointer is null".to_string(),
        ));
    }

    extern "C" {
        fn gst_app_src_end_of_stream(appsrc: *mut gst::ffi::GstElement) -> i32;
    }

    let ret = gst_app_src_end_of_stream(appsrc_ptr as *mut gst::ffi::GstElement);
    if ret != 0 {
        return Err(NvBufSurfaceError::BufferAcquisitionFailed(format!(
            "gst_app_src_end_of_stream returned {}",
            ret
        )));
    }

    Ok(())
}
