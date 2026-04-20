//! Per-class draw specs for the cars-demo pipeline.

use anyhow::Context;
use picasso::prelude::*;
use savant_core::draw::{
    BBoxSource, BoundingBoxDraw, ColorDraw, LabelDraw, LabelPosition, LabelPositionKind,
    ObjectDraw, PaddingDraw,
};
use savant_core::primitives::frame::VideoFrameProxy;
use savant_core::primitives::object::{IdCollisionResolutionPolicy, VideoObjectBuilder};
use savant_core::primitives::RBBox;

/// Namespace the draw spec keys under — matches
/// [`crate::cars_tracking::model::DETECTION_NAMESPACE`].
pub const DRAW_NAMESPACE: &str = crate::cars_tracking::model::DETECTION_NAMESPACE;

/// Namespace for non-detection overlay objects (e.g. the frame-id badge in
/// the top-left corner).  Kept distinct from [`DRAW_NAMESPACE`] so that the
/// tracker's batch formation — which filters by
/// [`crate::cars_tracking::model::DETECTION_NAMESPACE`] — never sees these
/// synthetic objects as ROIs.
pub const OVERLAY_NAMESPACE: &str = "overlay";

/// Fixed [`savant_core::primitives::object::VideoObject::label`] used for
/// the per-frame frame-id overlay.  Picasso keys the draw spec by
/// `(namespace, label)`, so this stays constant across frames; the
/// incrementing counter is stored in the object's `draw_label` field and
/// pulled into the rendered text via the `{draw_label}` template var.
pub const OVERLAY_FRAME_ID_LABEL: &str = "frame_id";

/// Per-class brand colors (R, G, B). Chosen for high contrast on urban scenes:
/// - car:        cyan-blue
/// - motorbike:  magenta
/// - bus:        amber
/// - truck:      lime-green
pub const COLOR_CAR: (i64, i64, i64) = (80, 200, 255);
pub const COLOR_MOTORBIKE: (i64, i64, i64) = (240, 60, 180);
pub const COLOR_BUS: (i64, i64, i64) = (255, 180, 40);
pub const COLOR_TRUCK: (i64, i64, i64) = (120, 255, 120);

/// Per-class draw with the label template preset.
fn build_vehicle_object_draw(color: (i64, i64, i64)) -> anyhow::Result<ObjectDraw> {
    let (r, g, b) = color;

    let border = ColorDraw::new(r, g, b, 255).context("failed to build bbox border color")?;
    let background =
        ColorDraw::new(r, g, b, 40).context("failed to build bbox background color")?;
    let bbox = BoundingBoxDraw::new(border, background, 2, PaddingDraw::default_padding())
        .context("failed to build bounding box draw")?;

    let font_color =
        ColorDraw::new(255, 255, 255, 255).context("failed to build label font color")?;
    let label_background =
        ColorDraw::new(r, g, b, 200).context("failed to build label background color")?;
    let label_border = ColorDraw::new(0, 0, 0, 0).context("failed to build label border color")?;
    let label_position = LabelPosition::new(LabelPositionKind::TopLeftOutside, 0, -2)
        .context("failed to build label position")?;
    let label_padding = PaddingDraw::new(4, 2, 4, 2).context("failed to build label padding")?;
    let label = LabelDraw::new(
        font_color,
        label_background,
        label_border,
        0.9,
        1,
        label_position,
        label_padding,
        vec![
            "{label} #{track_id}".into(),
            "object: #{id}".into(),
            "{confidence}".into(),
        ],
    )
    .context("failed to build label draw")?;

    Ok(ObjectDraw::with_bbox_source(
        Some(bbox),
        None,
        Some(label),
        false,
        BBoxSource::TrackingBox,
    ))
}

/// Build the `ObjectDrawSpec` for the cars-demo pipeline.
///
/// One entry per vehicle label under namespace `DRAW_NAMESPACE`; every entry
/// uses `BBoxSource::DetectionBox` (the detection box is always present;
/// `track_box` is only set once a tracker match has happened and Picasso
/// already deals correctly with the missing field).
///
/// Label format: `["{label} #{track_id}", "{confidence}"]`, outside top-left.
///
/// Returns `Result` because the `savant_core::draw` constructors can fail on
/// pathological input (they validate ranges); we propagate those errors as
/// `anyhow::Error`.
pub fn build_vehicle_draw_spec() -> anyhow::Result<picasso::ObjectDrawSpec> {
    let mut spec = ObjectDrawSpec::new();

    spec.insert(DRAW_NAMESPACE, "car", build_vehicle_object_draw(COLOR_CAR)?);
    spec.insert(
        DRAW_NAMESPACE,
        "motorbike",
        build_vehicle_object_draw(COLOR_MOTORBIKE)?,
    );
    spec.insert(DRAW_NAMESPACE, "bus", build_vehicle_object_draw(COLOR_BUS)?);
    spec.insert(
        DRAW_NAMESPACE,
        "truck",
        build_vehicle_object_draw(COLOR_TRUCK)?,
    );

    spec.insert(
        OVERLAY_NAMESPACE,
        OVERLAY_FRAME_ID_LABEL,
        build_frame_id_overlay_draw()?,
    );

    Ok(spec)
}

/// Overlay anchor box placed in the top-left of the frame.  The geometry is
/// independent of the input resolution because Picasso renders bboxes in
/// pixel space, not normalised coordinates.
const OVERLAY_ANCHOR_CX: f32 = 90.0;
const OVERLAY_ANCHOR_CY: f32 = 26.0;
const OVERLAY_ANCHOR_W: f32 = 80.0;
const OVERLAY_ANCHOR_H: f32 = 24.0;

/// Build the `ObjectDraw` for the frame-id overlay badge.
///
/// The badge is a solid dark rectangle anchored in the top-left of the
/// frame with white `"frame #<counter>"` text drawn inside it.  The
/// actual counter value is injected per-frame via the fake object's
/// `draw_label` field (pulled by the `{draw_label}` template var).
fn build_frame_id_overlay_draw() -> anyhow::Result<ObjectDraw> {
    let border = ColorDraw::new(255, 255, 255, 220).context("overlay bbox border color")?;
    let background = ColorDraw::new(0, 0, 0, 180).context("overlay bbox background color")?;
    let bbox = BoundingBoxDraw::new(border, background, 1, PaddingDraw::default_padding())
        .context("overlay bounding box draw")?;

    let font_color = ColorDraw::new(255, 255, 255, 255).context("overlay label font color")?;
    let label_background = ColorDraw::new(0, 0, 0, 0).context("overlay label background color")?;
    let label_border = ColorDraw::new(0, 0, 0, 0).context("overlay label border color")?;
    let label_position = LabelPosition::new(LabelPositionKind::TopLeftInside, 6, 4)
        .context("overlay label position")?;
    let label_padding = PaddingDraw::new(0, 0, 0, 0).context("overlay label padding")?;
    let label = LabelDraw::new(
        font_color,
        label_background,
        label_border,
        1.0,
        2,
        label_position,
        label_padding,
        vec!["frame #{draw_label}".into()],
    )
    .context("overlay label draw")?;

    Ok(ObjectDraw::with_bbox_source(
        Some(bbox),
        None,
        Some(label),
        false,
        BBoxSource::DetectionBox,
    ))
}

/// Attach the frame-id overlay object to `frame`.
///
/// The attached object uses [`OVERLAY_NAMESPACE`] / [`OVERLAY_FRAME_ID_LABEL`]
/// — keys registered by [`build_vehicle_draw_spec`].  The `frame_id`
/// counter is stored in the object's
/// [`VideoObject::draw_label`](savant_core::primitives::object::VideoObject)
/// field and rendered inside the badge via the `{draw_label}` template var.
///
/// The object carries
/// [`IdCollisionResolutionPolicy::GenerateNewId`] so it can coexist with
/// any detections already on the frame without ID clashes.
///
/// Called from the render stage (see
/// [`crate::cars_tracking::pipeline`]) because:
/// - inference's batch formation uses `RoiKind::FullFrame` so the overlay
///   is invisible there,
/// - tracker's batch formation filters by
///   [`crate::cars_tracking::model::DETECTION_NAMESPACE`] so the overlay
///   is invisible there too.
///
/// Returns an error only if the `VideoObject` fails to build (missing
/// required fields — all supplied here) or `add_object` rejects the
/// frame (shouldn't happen under `GenerateNewId`).
pub fn attach_frame_id_overlay(frame: &VideoFrameProxy, frame_id: u64) -> anyhow::Result<()> {
    let obj = VideoObjectBuilder::default()
        .id(0)
        .namespace(OVERLAY_NAMESPACE.to_string())
        .label(OVERLAY_FRAME_ID_LABEL.to_string())
        .draw_label(Some(frame_id.to_string()))
        .detection_box(RBBox::new(
            OVERLAY_ANCHOR_CX,
            OVERLAY_ANCHOR_CY,
            OVERLAY_ANCHOR_W,
            OVERLAY_ANCHOR_H,
            None,
        ))
        .build()
        .context("build frame-id overlay object")?;
    frame
        .add_object(obj, IdCollisionResolutionPolicy::GenerateNewId)
        .context("attach frame-id overlay to frame")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use savant_core::primitives::frame::{VideoFrameContent, VideoFrameTranscodingMethod};
    use savant_core::primitives::object::ObjectOperations;
    use savant_core::primitives::video_codec::VideoCodec;

    /// Spec must carry one entry per vehicle label the detector can emit —
    /// missing an entry silently drops those detections from the overlay.
    /// Same contract applies to the frame-id overlay entry.
    #[test]
    fn draw_spec_contains_all_expected_keys() -> anyhow::Result<()> {
        let spec = build_vehicle_draw_spec()?;
        for label in ["car", "motorbike", "bus", "truck"] {
            assert!(
                spec.lookup(DRAW_NAMESPACE, label).is_some(),
                "missing draw spec for {label}"
            );
        }
        assert!(
            spec.lookup(OVERLAY_NAMESPACE, OVERLAY_FRAME_ID_LABEL)
                .is_some(),
            "missing draw spec for frame-id overlay"
        );
        Ok(())
    }

    fn empty_frame() -> VideoFrameProxy {
        VideoFrameProxy::new(
            "test",
            (30, 1),
            1920,
            1080,
            VideoFrameContent::None,
            VideoFrameTranscodingMethod::Copy,
            Some(VideoCodec::H264),
            Some(true),
            (1, 1_000_000_000),
            0,
            None,
            None,
        )
        .expect("VideoFrameProxy::new")
    }

    #[test]
    fn attach_frame_id_overlay_adds_object_with_counter_as_draw_label() -> anyhow::Result<()> {
        let frame = empty_frame();
        attach_frame_id_overlay(&frame, 42)?;

        let objs = frame.get_all_objects();
        assert_eq!(objs.len(), 1, "exactly one overlay object expected");
        let obj = &objs[0];
        assert_eq!(obj.get_namespace(), OVERLAY_NAMESPACE);
        assert_eq!(obj.get_label(), OVERLAY_FRAME_ID_LABEL);
        assert_eq!(obj.get_draw_label(), Some("42".to_string()));
        Ok(())
    }

    #[test]
    fn attach_frame_id_overlay_reassigns_id_on_collision() -> anyhow::Result<()> {
        let frame = empty_frame();
        // Attach twice — the policy is `GenerateNewId`, so the second call
        // must not fail even though the builder always starts with id=0.
        attach_frame_id_overlay(&frame, 1)?;
        attach_frame_id_overlay(&frame, 2)?;

        let ids: std::collections::HashSet<i64> =
            frame.get_all_objects().iter().map(|o| o.get_id()).collect();
        assert_eq!(ids.len(), 2, "distinct ids assigned: {ids:?}");
        Ok(())
    }
}
