//! Attach `NvDsBatchMeta` with pre-tracker detections (`UNTRACKED_OBJECT_ID`).

use crate::error::{NvTrackerError, Result};
use crate::roi::Roi;
use deepstream_sys::{
    gst_buffer_add_nvds_meta, gst_buffer_get_nvds_batch_meta, nvds_acquire_frame_meta_from_pool,
    nvds_acquire_obj_meta_from_pool, nvds_add_frame_meta_to_batch, nvds_add_obj_meta_to_frame,
    nvds_batch_meta_copy_func, nvds_batch_meta_release_func, nvds_clear_obj_meta_list,
    nvds_create_batch_meta, nvds_meta_api_get_type, GstBuffer, GstNvDsMetaType_NVDS_BATCH_GST_META,
    NvDsBatchMeta, NvDsFrameMeta,
};
use savant_core::primitives::RBBox;
use std::ptr;
use std::sync::Once;

fn ensure_nvds_meta_api_registered() {
    static INIT: Once = Once::new();
    INIT.call_once(|| unsafe {
        nvds_meta_api_get_type();
    });
}

/// Pre-tracking placeholder (`UNTRACKED_OBJECT_ID` in DeepStream headers).
const PRETRACK_OBJECT_ID: u64 = u64::MAX;

fn rbbox_to_rect_unclamped(bbox: &RBBox, roi_id: i64, slot: u32) -> Result<(f32, f32, f32, f32)> {
    bbox.get_wrapping_bbox()
        .as_ltwh()
        .map_err(|e| NvTrackerError::BBoxConversion {
            roi_id,
            slot,
            reason: e.to_string(),
        })
}

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

/// Attach or update batch metadata with one frame per batch slot and detections per slot.
///
/// `slots[i]` is `(pad_index, classified_detections)` for surface slot `i` (`i` in `0..num_filled`).
/// Each detection is `(class_id, Roi)`.
/// `frame_nums[i]` is the DeepStream `frame_num` for that slot (global per-source counter).
/// Each [`Roi`] becomes an [`NvDsObjectMeta`] with `object_id = PRETRACK_OBJECT_ID`,
/// the given `class_id`, `confidence = 1.0`, and `misc_obj_info[0] = roi.id`.
/// Slots with an empty detection list get no object metas (allowed for shadow-track tests).
pub fn attach_detection_meta(
    buffer: &mut gstreamer::BufferRef,
    num_filled: u32,
    max_batch_size: u32,
    slots: &[(u32, Vec<(i32, Roi)>)],
    frame_nums: &[i32],
) -> Result<()> {
    if num_filled == 0 || num_filled > max_batch_size {
        return Err(NvTrackerError::batch_meta(
            "attach_detection_meta",
            format!("num_filled {num_filled} must be in 1..={max_batch_size}"),
        ));
    }
    if slots.len() != num_filled as usize {
        return Err(NvTrackerError::batch_meta(
            "attach_detection_meta",
            format!(
                "slots.len() {} must equal num_filled {}",
                slots.len(),
                num_filled
            ),
        ));
    }
    if frame_nums.len() != num_filled as usize {
        return Err(NvTrackerError::batch_meta(
            "attach_detection_meta",
            format!(
                "frame_nums.len() {} must equal num_filled {}",
                frame_nums.len(),
                num_filled
            ),
        ));
    }

    ensure_nvds_meta_api_registered();
    let buf_ptr = buffer.as_mut_ptr() as *mut GstBuffer;
    let existing: *mut NvDsBatchMeta = unsafe { gst_buffer_get_nvds_batch_meta(buf_ptr) };

    let batch_meta_raw: *mut NvDsBatchMeta = if !existing.is_null() {
        clear_frames_objects(existing);
        if unsafe { (*existing).num_frames_in_batch } != num_filled {
            return Err(NvTrackerError::batch_meta(
                "attach_detection_meta",
                format!(
                    "existing batch has {} frames but buffer num_filled is {}",
                    unsafe { (*existing).num_frames_in_batch },
                    num_filled
                ),
            ));
        }
        let mut frame_list = unsafe { (*existing).frame_meta_list };
        let mut i: u32 = 0;
        while !frame_list.is_null() && i < num_filled {
            let frame_ptr = unsafe { (*frame_list).data as *mut NvDsFrameMeta };
            if !frame_ptr.is_null() {
                let (pad_index, _) =
                    slots
                        .get(i as usize)
                        .ok_or(NvTrackerError::SlotIndexOutOfBounds {
                            index: i,
                            num_filled,
                            operation: "attach_detection_meta(patch frame list)".into(),
                        })?;
                let frame_num =
                    *frame_nums
                        .get(i as usize)
                        .ok_or(NvTrackerError::SlotIndexOutOfBounds {
                            index: i,
                            num_filled,
                            operation: "attach_detection_meta(frame_nums)".into(),
                        })?;
                unsafe {
                    (*frame_ptr).batch_id = i;
                    (*frame_ptr).frame_num = frame_num;
                    (*frame_ptr).pad_index = *pad_index;
                    (*frame_ptr).source_id = *pad_index;
                    (*frame_ptr).surface_index = 0;
                    (*frame_ptr).bInferDone = 1;
                }
                i += 1;
            }
            frame_list = unsafe { (*frame_list).next };
        }
        existing
    } else {
        let batch_meta = unsafe { nvds_create_batch_meta(max_batch_size) };
        if batch_meta.is_null() {
            return Err(NvTrackerError::NullPointer {
                function: "nvds_create_batch_meta".into(),
            });
        }
        for i in 0..num_filled {
            let frame_meta = unsafe { nvds_acquire_frame_meta_from_pool(batch_meta) };
            if frame_meta.is_null() {
                return Err(NvTrackerError::NullPointer {
                    function: "nvds_acquire_frame_meta_from_pool".into(),
                });
            }
            let (pad_index, _) =
                slots
                    .get(i as usize)
                    .ok_or(NvTrackerError::SlotIndexOutOfBounds {
                        index: i,
                        num_filled,
                        operation: "attach_detection_meta(create frames)".into(),
                    })?;
            let frame_num =
                *frame_nums
                    .get(i as usize)
                    .ok_or(NvTrackerError::SlotIndexOutOfBounds {
                        index: i,
                        num_filled,
                        operation: "attach_detection_meta(frame_nums create)".into(),
                    })?;
            unsafe {
                (*frame_meta).batch_id = i;
                (*frame_meta).frame_num = frame_num;
                (*frame_meta).pad_index = *pad_index;
                (*frame_meta).source_id = *pad_index;
                (*frame_meta).surface_index = 0;
                (*frame_meta).bInferDone = 1;
            }
            unsafe { nvds_add_frame_meta_to_batch(batch_meta, frame_meta) };
        }
        unsafe { (*batch_meta).num_frames_in_batch = num_filled };

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
            return Err(NvTrackerError::batch_meta(
                "attach_detection_meta",
                String::from("gst_buffer_add_nvds_meta failed"),
            ));
        }
        unsafe { (*added).meta_type = GstNvDsMetaType_NVDS_BATCH_GST_META };
        batch_meta
    };

    let mut frame_list = unsafe { (*batch_meta_raw).frame_meta_list };
    let mut slot: u32 = 0;
    while !frame_list.is_null() && slot < num_filled {
        let frame_ptr = unsafe { (*frame_list).data as *mut NvDsFrameMeta };
        if !frame_ptr.is_null() {
            add_slot_objects(batch_meta_raw, frame_ptr, slot, slots)?;
            slot += 1;
        }
        frame_list = unsafe { (*frame_list).next };
    }

    Ok(())
}

fn add_slot_objects(
    batch_meta: *mut NvDsBatchMeta,
    frame_meta: *mut NvDsFrameMeta,
    slot: u32,
    slots: &[(u32, Vec<(i32, Roi)>)],
) -> Result<()> {
    let (_, rois) = slots
        .get(slot as usize)
        .ok_or(NvTrackerError::SlotIndexOutOfBounds {
            index: slot,
            num_filled: slots.len() as u32,
            operation: "add_slot_objects".into(),
        })?;
    if rois.is_empty() {
        return Ok(());
    }
    let mut component_id = 1i32;
    for (class_id, roi) in rois.iter().rev() {
        let obj_meta = unsafe { nvds_acquire_obj_meta_from_pool(batch_meta) };
        if obj_meta.is_null() {
            return Err(NvTrackerError::NullPointer {
                function: "nvds_acquire_obj_meta_from_pool".into(),
            });
        }
        let (left, top, width, height) = rbbox_to_rect_unclamped(&roi.bbox, roi.id, slot)?;
        unsafe {
            (*obj_meta).rect_params.left = left;
            (*obj_meta).rect_params.top = top;
            (*obj_meta).rect_params.width = width;
            (*obj_meta).rect_params.height = height;
            (*obj_meta).detector_bbox_info.org_bbox_coords.left = left;
            (*obj_meta).detector_bbox_info.org_bbox_coords.top = top;
            (*obj_meta).detector_bbox_info.org_bbox_coords.width = width;
            (*obj_meta).detector_bbox_info.org_bbox_coords.height = height;
            (*obj_meta).object_id = PRETRACK_OBJECT_ID;
            (*obj_meta).class_id = *class_id;
            (*obj_meta).confidence = 1.0;
            (*obj_meta).unique_component_id = component_id;
            component_id = component_id.saturating_add(1);
            let mut misc = (*obj_meta).misc_obj_info;
            misc[0] = roi.id;
            (*obj_meta).misc_obj_info = misc;
            nvds_add_obj_meta_to_frame(frame_meta, obj_meta, ptr::null_mut());
        }
    }
    Ok(())
}
