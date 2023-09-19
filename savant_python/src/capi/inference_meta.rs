use crate::primitives::objects_view::VideoObjectsView;

#[repr(C)]
pub enum BoxSource {
    Detection = 0,
    Tracking = 1,
    TrackingWhenAbsentDetection = 2,
}

#[repr(C)]
pub struct InferenceMeta {
    pub id: i64,
    pub parent_id: i64,
    pub namespace: i64,
    pub label: i64,
    pub left: i64,
    pub top: i64,
    pub width: i64,
    pub height: i64,
}

/// # Safety
///
/// The function is unsafe because it is exported to C-ABI and works with raw pointers.
///
#[no_mangle]
pub unsafe extern "C" fn build_inference_meta(
    handle: usize,
    box_source: BoxSource,
    meta: *mut InferenceMeta,
    meta_capacity: usize,
) -> usize {
    let video_object_view = &*(handle as *const VideoObjectsView);
    if meta_capacity < video_object_view.len() {
        panic!("max_elts is less than the number of objects in the view");
    }

    for (index, o) in video_object_view.inner.iter().enumerate() {
        let bb = match box_source {
            BoxSource::Detection => Some(o.get_detection_box()),
            BoxSource::Tracking => o.get_track_box(),
            BoxSource::TrackingWhenAbsentDetection => {
                o.get_track_box().or(Some(o.get_detection_box()))
            }
        };
        if bb.is_none() {
            panic!(
                "Box source is not available for object with Id {}",
                o.get_id()
            );
        }

        let bb = bb.unwrap();
        if bb.get_angle().unwrap_or(0.0) != 0.0 {
            panic!("Only objects with boxes without an angle or with an angle equal to 0.0 can be passed to the inference. Received: {:?}, Object Id: {}", &bb, o.get_id());
        }

        let (left, top, width, height) = bb.as_ltrb_int().unwrap_or_else(|e| {
            panic!(
                "Failed to get the LTRB representation for a box: {:?}, Object Id: {}, Err: {:?}",
                &bb,
                o.get_id(),
                e
            )
        });

        let inner_obj = &o.0;
        let meta_item = &mut *meta.add(index);

        meta_item.id = inner_obj.get_id();
        meta_item.parent_id = inner_obj.get_parent_id().unwrap_or(i64::MAX);
        meta_item.namespace = inner_obj.get_namespace_id().unwrap_or(i64::MAX);
        meta_item.label = inner_obj.get_label_id().unwrap_or(i64::MAX);
        meta_item.left = left;
        meta_item.top = top;
        meta_item.width = width;
        meta_item.height = height;
    }
    video_object_view.len()
}
