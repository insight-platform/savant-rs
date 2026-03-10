//! Safe NvDsBatchMeta creation and attachment to gst::Buffer.
//!
//! The primary entry-point is [`attach_batch_meta_with_rois`], which:
//!
//! 1. Optionally **clears** existing [`NvDsObjectMeta`] entries from each
//!    frame if the buffer already carries a batch meta (see
//!    [`MetaClearPolicy`]).
//! 2. Creates a new batch meta (or reuses the existing one) and populates
//!    frame metas for every batch slot.
//! 3. Adds one [`NvDsObjectMeta`] per ROI to each slot's frame meta so that
//!    `Gst-nvinfer` in secondary (`process-mode=2`) mode processes each ROI
//!    independently.
//!    When no ROIs are supplied for a slot a single full-frame sentinel object
//!    (`unique_component_id = FULL_FRAME_SENTINEL`) is added instead.
//!
//! A complementary helper [`clear_all_frame_objects`] removes every
//! [`NvDsObjectMeta`] from all frames in an existing batch meta; it is called
//! by [`crate::output::BatchInferenceOutput`] on drop when the policy is
//! [`MetaClearPolicy::After`] or [`MetaClearPolicy::Both`].

use crate::error::{NvInferError, Result};
use crate::meta_clear_policy::MetaClearPolicy;
use crate::roi::Roi;
use deepstream_sys::{
    gst_buffer_add_nvds_meta, gst_buffer_get_nvds_batch_meta, nvds_acquire_frame_meta_from_pool,
    nvds_acquire_obj_meta_from_pool, nvds_add_frame_meta_to_batch, nvds_add_obj_meta_to_frame,
    nvds_batch_meta_copy_func, nvds_batch_meta_release_func, nvds_clear_obj_meta_list,
    nvds_create_batch_meta, nvds_meta_api_get_type, GstBuffer, GstNvDsMetaType_NVDS_BATCH_GST_META,
    NvDsBatchMeta, NvDsFrameMeta,
};
use savant_core::primitives::RBBox;
use std::collections::HashMap;
use std::ptr;
use std::sync::Once;

/// Force-initialise the private tag quark inside `libnvdsgst_meta.so`.
///
/// `gst_buffer_get_nvds_batch_meta` reads a file-scoped GQuark that is only
/// set as a side-effect of `nvds_meta_api_get_type()`.  If the API type has
/// never been registered the quark is still zero (BSS), which triggers:
///
/// ```text
/// GStreamer-CRITICAL **: gst_meta_api_type_has_tag: assertion 'tag != 0' failed
/// ```
fn ensure_nvds_meta_api_registered() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        unsafe { nvds_meta_api_get_type() };
    });
}

/// Sentinel `unique_component_id` written to full-frame synthetic objects
/// (no explicit ROI supplied for a slot). Callers can inspect this value in
/// [`crate::output::ElementOutput::roi_id`] to distinguish full-frame results
/// from explicit-ROI results (`roi_id` will be `None`).
pub(crate) const FULL_FRAME_SENTINEL: i32 = -1;

/// Attach (or update) `NvDsBatchMeta` on `buffer` for secondary-mode ROI
/// inference, honouring `policy` for object meta clearing.
///
/// ## Clearing (Before)
/// When `policy` is [`MetaClearPolicy::Before`] or [`MetaClearPolicy::Both`]
/// and `buffer` already carries a `NvDsBatchMeta`, **all object metadata is
/// cleared** from every frame via NVIDIA's `nvds_clear_obj_meta_list` before
/// new ROI objects are added.  This prevents stale objects from previous
/// pipeline stages from interfering with inference.
///
/// When `policy` is [`MetaClearPolicy::None`] or [`MetaClearPolicy::After`]
/// the existing objects are **not** cleared.
///
/// ## ROI objects
/// For each of the `num_frames` batch slots:
/// * If `rois` contains a non-empty list for that slot → one
///   `NvDsObjectMeta` per [`Roi`] is added.  `object_id` is set to
///   `roi.id as u64` so the caller can recover `roi_id` from the output
///   tensor meta.
/// * Otherwise → a single full-frame sentinel object covering
///   `(0, 0, input_width, input_height)` is added with
///   `unique_component_id = FULL_FRAME_SENTINEL`.
///
/// ## Arguments
/// * `buffer` – Writable `gst::BufferRef` that must already contain an
///   `NvBufSurface`.
/// * `num_frames` – Number of filled slots (`numFilled` from the surface).
/// * `max_batch_size` – Pool size for the newly created meta (ignored when an
///   existing meta is reused).
/// * `policy` – When to clear existing object metas.
/// * `rois` – Optional per-slot ROI lists (key = slot index `0..num_frames`).
/// * `input_width` / `input_height` – Model input dimensions used for the
///   full-frame fallback.
pub fn attach_batch_meta_with_rois(
    buffer: &mut gstreamer::BufferRef,
    num_frames: u32,
    max_batch_size: u32,
    policy: MetaClearPolicy,
    rois: Option<&HashMap<u32, Vec<Roi>>>,
    input_width: u32,
    input_height: u32,
) -> Result<()> {
    if num_frames == 0 || num_frames > max_batch_size {
        return Err(NvInferError::BatchMetaFailed(format!(
            "num_frames {num_frames} must be in 1..={max_batch_size}"
        )));
    }

    ensure_nvds_meta_api_registered();

    let buf_ptr = buffer.as_mut_ptr() as *mut GstBuffer;

    // Check whether the buffer already has a batch meta.
    let existing: *mut NvDsBatchMeta = unsafe { gst_buffer_get_nvds_batch_meta(buf_ptr) };

    let batch_meta_raw: *mut NvDsBatchMeta = if !existing.is_null() {
        if policy.clear_before() {
            // Clear all existing object metas from every frame so stale objects
            // from previous pipeline stages do not reach nvinfer.
            clear_frames_objects(existing);
        }
        existing
    } else {
        // Allocate a fresh batch meta and populate one frame meta per slot.
        let batch_meta = unsafe { nvds_create_batch_meta(max_batch_size) };
        if batch_meta.is_null() {
            return Err(NvInferError::NullPointer("nvds_create_batch_meta".into()));
        }

        for i in 0..num_frames {
            let frame_meta = unsafe { nvds_acquire_frame_meta_from_pool(batch_meta) };
            if frame_meta.is_null() {
                return Err(NvInferError::NullPointer(
                    "nvds_acquire_frame_meta_from_pool".into(),
                ));
            }
            unsafe {
                (*frame_meta).batch_id = i;
                (*frame_meta).frame_num = i as i32;
                // All frames share pad 0 (the single appsrc pad).  Setting
                // pad_index > 0 would cause Gst-nvinfer to skip the frame
                // with "Source info not found for source N".
                (*frame_meta).pad_index = 0;
                (*frame_meta).source_id = 0;
                (*frame_meta).source_frame_width = input_width;
                (*frame_meta).source_frame_height = input_height;
            }
            unsafe { nvds_add_frame_meta_to_batch(batch_meta, frame_meta) };
        }
        unsafe { (*batch_meta).num_frames_in_batch = num_frames };

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
            return Err(NvInferError::BatchMetaFailed(
                "gst_buffer_add_nvds_meta failed".into(),
            ));
        }
        unsafe { (*added).meta_type = GstNvDsMetaType_NVDS_BATCH_GST_META };

        batch_meta
    };

    // Add ROI object metas to each frame in list traversal order (matches
    // the slot order we set via batch_id when creating).
    let mut frame_list = unsafe { (*batch_meta_raw).frame_meta_list };
    let mut slot: u32 = 0;
    while !frame_list.is_null() && slot < num_frames {
        let frame_ptr = unsafe { (*frame_list).data as *mut NvDsFrameMeta };
        if !frame_ptr.is_null() {
            add_roi_objects(
                batch_meta_raw,
                frame_ptr,
                slot,
                rois,
                input_width,
                input_height,
            );
            slot += 1;
        }
        frame_list = unsafe { (*frame_list).next };
    }

    Ok(())
}

/// Remove every `NvDsObjectMeta` from every frame in the batch meta attached
/// to `buf_ptr`.
///
/// This is the **after** clearing path called by
/// [`crate::output::BatchInferenceOutput::drop`] when the policy includes
/// [`MetaClearPolicy::After`].
///
/// # Safety
/// `buf_ptr` must be a valid, writable `*mut GstBuffer`.
pub(crate) unsafe fn clear_all_frame_objects(buf_ptr: *mut GstBuffer) {
    ensure_nvds_meta_api_registered();
    let batch_meta = gst_buffer_get_nvds_batch_meta(buf_ptr);
    if batch_meta.is_null() {
        return;
    }
    clear_frames_objects(batch_meta);
}

/// Iterate the frame meta list of `batch_meta` and call
/// `nvds_clear_obj_meta_list` on each non-empty frame.
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

/// Convert an [`RBBox`] to axis-aligned `(left, top, width, height)` suitable
/// for `NvDsObjectMeta.rect_params`.
///
/// For axis-aligned boxes the conversion is a straightforward center-to-ltwh.
/// For rotated boxes `get_wrapping_bbox` produces the tight axis-aligned
/// envelope; both paths then clamp to `[0, max_w] × [0, max_h]`.
fn rbbox_to_rect_params(bbox: &RBBox, max_w: f32, max_h: f32) -> (f32, f32, f32, f32) {
    let is_rotated = bbox.get_angle().is_some_and(|a| a.abs() > f32::EPSILON);

    let (left, top, w, h) = if is_rotated {
        // wrapping bbox is always axis-aligned, as_ltwh always succeeds
        bbox.get_wrapping_bbox().as_ltwh().unwrap()
    } else {
        bbox.as_ltwh().unwrap()
    };

    if max_w > 0.0 && max_h > 0.0 {
        let left_clamped = left.max(0.0);
        let top_clamped = top.max(0.0);
        let right_clamped = (left + w).min(max_w);
        let bottom_clamped = (top + h).min(max_h);
        let clamped_w = (right_clamped - left_clamped).max(1.0);
        let clamped_h = (bottom_clamped - top_clamped).max(1.0);
        (left_clamped, top_clamped, clamped_w, clamped_h)
    } else {
        (left, top, w, h)
    }
}

/// Add `NvDsObjectMeta` entries for the given slot to `frame_ptr`.
///
/// Uses explicit ROIs when present, or a single full-frame sentinel otherwise.
fn add_roi_objects(
    batch_meta: *mut NvDsBatchMeta,
    frame_meta: *mut NvDsFrameMeta,
    slot: u32,
    rois: Option<&HashMap<u32, Vec<Roi>>>,
    input_width: u32,
    input_height: u32,
) {
    let slot_rois = rois.and_then(|r| r.get(&slot));
    let max_w = input_width as f32;
    let max_h = input_height as f32;

    if let Some(roi_list) = slot_rois.filter(|v| !v.is_empty()) {
        // Iterate in reverse because nvds_add_obj_meta_to_frame prepends
        // to the GList, so the first ROI in the input vec ends up first
        // when the list is traversed head-to-tail during output extraction.
        for roi in roi_list.iter().rev() {
            let obj_meta = unsafe { nvds_acquire_obj_meta_from_pool(batch_meta) };
            if obj_meta.is_null() {
                log::error!("nvds_acquire_obj_meta_from_pool returned null for slot {slot}");
                continue;
            }
            let (left, top, width, height) = rbbox_to_rect_params(&roi.bbox, max_w, max_h);
            unsafe {
                (*obj_meta).rect_params.left = left;
                (*obj_meta).rect_params.top = top;
                (*obj_meta).rect_params.width = width;
                (*obj_meta).rect_params.height = height;
                (*obj_meta).object_id = roi.id as u64;
                (*obj_meta).unique_component_id = 0;
                nvds_add_obj_meta_to_frame(frame_meta, obj_meta, ptr::null_mut());
            }
        }
    } else {
        // No explicit ROIs – synthesise a full-frame object so that nvinfer
        // in secondary mode still receives a region to process.
        let obj_meta = unsafe { nvds_acquire_obj_meta_from_pool(batch_meta) };
        if obj_meta.is_null() {
            log::error!(
                "nvds_acquire_obj_meta_from_pool returned null (full-frame) for slot {slot}"
            );
            return;
        }
        unsafe {
            (*obj_meta).rect_params.left = 0.0;
            (*obj_meta).rect_params.top = 0.0;
            (*obj_meta).rect_params.width = input_width as f32;
            (*obj_meta).rect_params.height = input_height as f32;
            (*obj_meta).unique_component_id = FULL_FRAME_SENTINEL;
            nvds_add_obj_meta_to_frame(frame_meta, obj_meta, ptr::null_mut());
        }
    }
}

// ─── helpers used by tests ────────────────────────────────────────────────────

/// Count the number of `NvDsObjectMeta` entries attached to the first frame of
/// the batch meta on `buf_ptr`. Returns 0 if no batch meta or no frames.
#[cfg(test)]
pub(crate) unsafe fn count_objects_in_first_frame(buf_ptr: *mut GstBuffer) -> u32 {
    ensure_nvds_meta_api_registered();
    let batch_meta = gst_buffer_get_nvds_batch_meta(buf_ptr);
    if batch_meta.is_null() {
        return 0;
    }
    let frame_list = (*batch_meta).frame_meta_list;
    if frame_list.is_null() {
        return 0;
    }
    let frame_ptr = (*frame_list).data as *mut NvDsFrameMeta;
    if frame_ptr.is_null() {
        return 0;
    }
    let mut obj_list = (*frame_ptr).obj_meta_list;
    let mut count = 0u32;
    while !obj_list.is_null() {
        if !(*obj_list).data.is_null() {
            count += 1;
        }
        obj_list = (*obj_list).next;
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use deepstream_sys::{
        nvds_acquire_frame_meta_from_pool, nvds_add_frame_meta_to_batch, nvds_create_batch_meta,
    };

    // ── shared fixture ───────────────────────────────────────────────────────

    /// Build a minimal GStreamer buffer that looks like a 1-slot NvBufSurface.
    fn make_surface_buffer(batch_size: u32, num_filled: u32) -> gstreamer::Buffer {
        let mut header = [0u8; 12];
        header[0..4].copy_from_slice(&0u32.to_ne_bytes()); // gpuId
        header[4..8].copy_from_slice(&batch_size.to_ne_bytes());
        header[8..12].copy_from_slice(&num_filled.to_ne_bytes());
        gstreamer::Buffer::from_slice(header).copy()
    }

    /// Attach a batch meta with `n_objects` dummy objects to the first frame.
    unsafe fn attach_preexisting_batch_meta(buf: &mut gstreamer::Buffer, n_objects: u32) {
        let buf_ptr = buf.get_mut().unwrap().as_mut_ptr() as *mut GstBuffer;

        let batch_meta = nvds_create_batch_meta(1);
        let frame_meta = nvds_acquire_frame_meta_from_pool(batch_meta);
        (*frame_meta).batch_id = 0;
        (*frame_meta).frame_num = 0;
        (*frame_meta).pad_index = 0;
        nvds_add_frame_meta_to_batch(batch_meta, frame_meta);
        (*batch_meta).num_frames_in_batch = 1;

        for id in 0..n_objects {
            let obj = nvds_acquire_obj_meta_from_pool(batch_meta);
            (*obj).object_id = id as u64 + 100; // id 100, 101, … to distinguish from ROI ids
            nvds_add_obj_meta_to_frame(frame_meta, obj, ptr::null_mut());
        }

        let added = gst_buffer_add_nvds_meta(
            buf_ptr,
            batch_meta as *mut _,
            ptr::null_mut(),
            Some(nvds_batch_meta_copy_func),
            Some(nvds_batch_meta_release_func),
        );
        (*added).meta_type = GstNvDsMetaType_NVDS_BATCH_GST_META;
    }

    fn roi_42() -> Roi {
        Roi {
            id: 42,
            bbox: RBBox::ltwh(0.0, 0.0, 64.0, 64.0).unwrap(),
        }
    }

    fn rois_for_slot0(roi: Roi) -> HashMap<u32, Vec<Roi>> {
        [(0, vec![roi])].into()
    }

    // ── tests ────────────────────────────────────────────────────────────────

    /// `MetaClearPolicy::Before` clears stale objects before adding ROI ones.
    #[test]
    fn policy_before_clears_existing_objects() {
        let _ = gstreamer::init();

        let mut buf = make_surface_buffer(1, 1);
        // Pre-populate 2 stale objects.
        unsafe { attach_preexisting_batch_meta(&mut buf, 2) };

        let rois = rois_for_slot0(roi_42());
        let buf_ref = buf.get_mut().unwrap();
        attach_batch_meta_with_rois(buf_ref, 1, 1, MetaClearPolicy::Before, Some(&rois), 64, 64)
            .unwrap();

        let count = unsafe { count_objects_in_first_frame(buf_ref.as_mut_ptr() as *mut GstBuffer) };
        assert_eq!(
            count, 1,
            "stale objects must be cleared; only ROI object remains"
        );
    }

    /// `MetaClearPolicy::None` preserves existing objects; new ROI is appended.
    #[test]
    fn policy_none_keeps_existing_objects() {
        let _ = gstreamer::init();

        let mut buf = make_surface_buffer(1, 1);
        unsafe { attach_preexisting_batch_meta(&mut buf, 2) };

        let rois = rois_for_slot0(roi_42());
        let buf_ref = buf.get_mut().unwrap();
        attach_batch_meta_with_rois(buf_ref, 1, 1, MetaClearPolicy::None, Some(&rois), 64, 64)
            .unwrap();

        let count = unsafe { count_objects_in_first_frame(buf_ref.as_mut_ptr() as *mut GstBuffer) };
        // 2 pre-existing + 1 ROI = 3
        assert_eq!(
            count, 3,
            "None policy must not clear; total should be stale + new ROI"
        );
    }

    /// `MetaClearPolicy::After` does NOT clear existing objects at submit time.
    #[test]
    fn policy_after_does_not_clear_before() {
        let _ = gstreamer::init();

        let mut buf = make_surface_buffer(1, 1);
        unsafe { attach_preexisting_batch_meta(&mut buf, 2) };

        let rois = rois_for_slot0(roi_42());
        let buf_ref = buf.get_mut().unwrap();
        attach_batch_meta_with_rois(buf_ref, 1, 1, MetaClearPolicy::After, Some(&rois), 64, 64)
            .unwrap();

        let count = unsafe { count_objects_in_first_frame(buf_ref.as_mut_ptr() as *mut GstBuffer) };
        // After policy clears on drop, not on submit → 2 + 1 = 3
        assert_eq!(count, 3, "After policy must not clear at submit time");
    }

    /// `MetaClearPolicy::Both` clears before (same as Before).
    #[test]
    fn policy_both_clears_before() {
        let _ = gstreamer::init();

        let mut buf = make_surface_buffer(1, 1);
        unsafe { attach_preexisting_batch_meta(&mut buf, 3) };

        let rois = rois_for_slot0(roi_42());
        let buf_ref = buf.get_mut().unwrap();
        attach_batch_meta_with_rois(buf_ref, 1, 1, MetaClearPolicy::Both, Some(&rois), 64, 64)
            .unwrap();

        let count = unsafe { count_objects_in_first_frame(buf_ref.as_mut_ptr() as *mut GstBuffer) };
        assert_eq!(
            count, 1,
            "Both policy must clear stale objects before submit"
        );
    }

    /// `clear_all_frame_objects` removes every object from all frames.
    #[test]
    fn clear_all_frame_objects_empties_frames() {
        let _ = gstreamer::init();

        let mut buf = make_surface_buffer(1, 1);
        unsafe { attach_preexisting_batch_meta(&mut buf, 4) };

        {
            let buf_ref = buf.get_mut().unwrap();
            let count =
                unsafe { count_objects_in_first_frame(buf_ref.as_mut_ptr() as *mut GstBuffer) };
            assert_eq!(count, 4, "precondition: 4 objects must be present");

            unsafe { clear_all_frame_objects(buf_ref.as_mut_ptr() as *mut GstBuffer) };

            let after =
                unsafe { count_objects_in_first_frame(buf_ref.as_mut_ptr() as *mut GstBuffer) };
            assert_eq!(after, 0, "all objects must be removed after clear");
        }
    }

    #[test]
    fn rbbox_to_rect_params_axis_aligned() {
        let bbox = RBBox::new(50.0, 40.0, 20.0, 30.0, None);
        let (l, t, w, h) = rbbox_to_rect_params(&bbox, 200.0, 200.0);
        assert!((l - 40.0).abs() < 0.01);
        assert!((t - 25.0).abs() < 0.01);
        assert!((w - 20.0).abs() < 0.01);
        assert!((h - 30.0).abs() < 0.01);
    }

    #[test]
    fn rbbox_to_rect_params_axis_aligned_zero_angle() {
        let bbox = RBBox::new(50.0, 40.0, 20.0, 30.0, Some(0.0));
        let (l, t, w, h) = rbbox_to_rect_params(&bbox, 200.0, 200.0);
        assert!((l - 40.0).abs() < 0.01);
        assert!((t - 25.0).abs() < 0.01);
        assert!((w - 20.0).abs() < 0.01);
        assert!((h - 30.0).abs() < 0.01);
    }

    #[test]
    fn rbbox_to_rect_params_rotated_45() {
        let bbox = RBBox::new(100.0, 100.0, 50.0, 50.0, Some(45.0));
        let (l, t, w, h) = rbbox_to_rect_params(&bbox, 500.0, 500.0);
        // 45-degree rotation of a 50x50 box: wrapping box ≈ 70.71 × 70.71
        let diag = 50.0 * std::f32::consts::SQRT_2;
        let half_diag = diag / 2.0;
        // Must clamp to [0, 500], not [2, 498] — no rendering inset.
        assert!(
            (l - (100.0 - half_diag)).abs() < 1.0,
            "left ≈ {}, got {l}",
            100.0 - half_diag
        );
        assert!(
            (t - (100.0 - half_diag)).abs() < 1.0,
            "top ≈ {}, got {t}",
            100.0 - half_diag
        );
        assert!((w - diag).abs() < 1.0, "wrapping width ≈ {diag}, got {w}");
        assert!((h - diag).abs() < 1.0, "wrapping height ≈ {diag}, got {h}");
    }

    #[test]
    fn rbbox_to_rect_params_clamped_to_frame() {
        let bbox = RBBox::new(5.0, 5.0, 20.0, 20.0, None);
        let (l, t, w, h) = rbbox_to_rect_params(&bbox, 15.0, 15.0);
        assert!(l >= 0.0, "left must not be negative");
        assert!(t >= 0.0, "top must not be negative");
        assert!(l + w <= 15.0 + 0.01, "right edge must not exceed max_w");
        assert!(t + h <= 15.0 + 0.01, "bottom edge must not exceed max_h");
    }

    /// Rotated box near the left/top edge: must clamp to 0, not to 2.
    #[test]
    fn rbbox_to_rect_params_rotated_near_origin_clamps_to_zero() {
        // 50×50 box centered at (25,25) rotated 45°.
        // Wrapping bbox: ~70.71×70.71 centered at (25,25).
        // Left ≈ 25 − 35.35 ≈ −10.35, clamped to 0.
        // Top  ≈ 25 − 35.35 ≈ −10.35, clamped to 0.
        let bbox = RBBox::new(25.0, 25.0, 50.0, 50.0, Some(45.0));
        let (l, t, _w, _h) = rbbox_to_rect_params(&bbox, 500.0, 500.0);
        assert!(
            l.abs() < 0.01,
            "rotated box near origin: left must clamp to 0, got {l}"
        );
        assert!(
            t.abs() < 0.01,
            "rotated box near origin: top must clamp to 0, got {t}"
        );
    }

    /// Rotated box near the right/bottom edge: must clamp to max_w/max_h,
    /// not to max−2.
    #[test]
    fn rbbox_to_rect_params_rotated_near_far_edge_clamps_to_max() {
        // 50×50 box centered at (475,475) rotated 45° in a 500×500 frame.
        // Wrapping bbox: ~70.71×70.71 centered at (475,475).
        // Right ≈ 475 + 35.35 ≈ 510.35, clamped to 500.
        // Bottom ≈ 475 + 35.35 ≈ 510.35, clamped to 500.
        let bbox = RBBox::new(475.0, 475.0, 50.0, 50.0, Some(45.0));
        let (l, t, w, h) = rbbox_to_rect_params(&bbox, 500.0, 500.0);
        let right = l + w;
        let bottom = t + h;
        assert!(
            (right - 500.0).abs() < 0.01,
            "rotated box near far edge: right must clamp to 500, got {right}"
        );
        assert!(
            (bottom - 500.0).abs() < 0.01,
            "rotated box near far edge: bottom must clamp to 500, got {bottom}"
        );
    }

    /// Rotated and axis-aligned boxes at the same position must produce
    /// consistent clamping (both use [0, max] bounds, no asymmetric inset).
    #[test]
    fn rbbox_to_rect_params_rotated_vs_axis_aligned_same_clamping() {
        // Axis-aligned box touching left edge: left=0, w=40, in a 100×100 frame.
        let aa = RBBox::ltwh(0.0, 0.0, 40.0, 40.0).unwrap();
        let (l_aa, t_aa, w_aa, h_aa) = rbbox_to_rect_params(&aa, 100.0, 100.0);

        // Same box but with angle=0.001 (just above EPSILON, triggers rotated path).
        let rot = RBBox::new(20.0, 20.0, 40.0, 40.0, Some(0.001));
        let (l_rot, t_rot, w_rot, h_rot) = rbbox_to_rect_params(&rot, 100.0, 100.0);

        assert!(
            (l_aa - l_rot).abs() < 1.0,
            "left: axis-aligned={l_aa}, rotated={l_rot}"
        );
        assert!(
            (t_aa - t_rot).abs() < 1.0,
            "top: axis-aligned={t_aa}, rotated={t_rot}"
        );
        assert!(
            (w_aa - w_rot).abs() < 1.0,
            "width: axis-aligned={w_aa}, rotated={w_rot}"
        );
        assert!(
            (h_aa - h_rot).abs() < 1.0,
            "height: axis-aligned={h_aa}, rotated={h_rot}"
        );
    }

    /// When left/top is negative and clamped to 0, width/height must reflect
    /// the actual visible extent, not the original unclamped dimensions.
    #[test]
    fn rbbox_to_rect_params_negative_left_top_reduces_visible_size() {
        // Box from left=-5 to right=15 (w=20), top=0 to bottom=10 (h=10).
        // Visible: left clamped to 0, so visible region is 0..15 x 0..10.
        // Width must be 15, not 20.
        let bbox = RBBox::ltwh(-5.0, 0.0, 20.0, 10.0).unwrap();
        let (l, t, w, h) = rbbox_to_rect_params(&bbox, 100.0, 100.0);
        assert!((l - 0.0).abs() < 0.01, "left must be clamped to 0");
        assert!((t - 0.0).abs() < 0.01, "top unchanged");
        assert!(
            (w - 15.0).abs() < 0.01,
            "visible width must be 15 (0 to 15), not original 20"
        );
        assert!((h - 10.0).abs() < 0.01, "height unchanged");

        // Box with negative top: top=-3, bottom=7 (h=10). Visible 0..10 x 0..7.
        let bbox = RBBox::ltwh(0.0, -3.0, 10.0, 10.0).unwrap();
        let (l, t, w, h) = rbbox_to_rect_params(&bbox, 100.0, 100.0);
        assert!((l - 0.0).abs() < 0.01);
        assert!((t - 0.0).abs() < 0.01, "top must be clamped to 0");
        assert!((w - 10.0).abs() < 0.01);
        assert!(
            (h - 7.0).abs() < 0.01,
            "visible height must be 7 (0 to 7), not original 10"
        );
    }
}
