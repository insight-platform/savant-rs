//! Safe NvDsBatchMeta creation and attachment to gst::Buffer.

use crate::error::{Result, SidecarError};
use deepstream_sys::{
    gst_buffer_add_nvds_meta, nvds_acquire_frame_meta_from_pool, nvds_add_frame_meta_to_batch,
    nvds_batch_meta_copy_func, nvds_batch_meta_release_func, nvds_create_batch_meta, GstBuffer,
    GstNvDsMetaType_NVDS_BATCH_GST_META,
};
#[allow(unused_imports)]
use gstreamer::prelude::*;
use std::ptr;

/// Attach NvDsBatchMeta to a gst::Buffer that already has a batched NvBufSurface.
///
/// Creates batch meta with `max_batch_size` slots, populates `num_frames` frame
/// metas (one per batch element), and attaches to the buffer.
///
/// # Arguments
/// * `buffer` - The buffer to attach batch meta to (must have NvBufSurface).
/// * `num_frames` - Number of frames in the batch (numFilled from NvBufSurface).
/// * `max_batch_size` - Maximum batch size for the meta structure.
pub fn attach_batch_meta(
    buffer: &mut gstreamer::BufferRef,
    num_frames: u32,
    max_batch_size: u32,
) -> Result<()> {
    if num_frames == 0 || num_frames > max_batch_size {
        return Err(SidecarError::BatchMetaFailed(format!(
            "num_frames {} must be in 1..={}",
            num_frames, max_batch_size
        )));
    }

    let batch_meta = unsafe { nvds_create_batch_meta(max_batch_size) };
    if batch_meta.is_null() {
        return Err(SidecarError::NullPointer("nvds_create_batch_meta".into()));
    }

    // Populate frame metas for each batch element.
    for i in 0..num_frames {
        let frame_meta = unsafe { nvds_acquire_frame_meta_from_pool(batch_meta) };
        if frame_meta.is_null() {
            return Err(SidecarError::NullPointer(
                "nvds_acquire_frame_meta_from_pool".into(),
            ));
        }
        unsafe {
            (*frame_meta).batch_id = 0;
            (*frame_meta).frame_num = i as i32;
            (*frame_meta).pad_index = i;
            (*frame_meta).source_id = 0;
        }
        unsafe { nvds_add_frame_meta_to_batch(batch_meta, frame_meta) };
    }

    unsafe {
        (*batch_meta).num_frames_in_batch = num_frames;
    }

    let buf_ptr = buffer.as_mut_ptr() as *mut GstBuffer;
    let added = unsafe {
        gst_buffer_add_nvds_meta(
            buf_ptr,
            batch_meta as *mut _,
            ptr::null_mut(),
            Some(nvds_batch_meta_copy_func),
            Some(nvds_batch_meta_release_func),
        )
    };

    if added.is_null() {
        return Err(SidecarError::BatchMetaFailed(
            "gst_buffer_add_nvds_meta failed".into(),
        ));
    }

    // Mark this NvDsMeta as batch meta so gst_buffer_get_nvds_batch_meta() can find it.
    unsafe {
        (*added).meta_type = GstNvDsMetaType_NVDS_BATCH_GST_META;
    }

    Ok(())
}
