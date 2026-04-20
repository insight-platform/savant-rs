//! Per-class draw specs for the cars-demo pipeline.

use anyhow::Context;
use picasso::prelude::*;
use savant_core::draw::{
    BBoxSource, BoundingBoxDraw, ColorDraw, LabelDraw, LabelPosition, LabelPositionKind,
    ObjectDraw, PaddingDraw,
};

/// Namespace the draw spec keys under — matches
/// [`crate::cars_tracking::model::DETECTION_NAMESPACE`].
pub const DRAW_NAMESPACE: &str = crate::cars_tracking::model::DETECTION_NAMESPACE;

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
        vec!["{label} #{track_id}".into(), "object: #{id}".into(), "{confidence}".into()],
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

    Ok(spec)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Spec must carry one entry per vehicle label the detector can emit —
    /// missing an entry silently drops those detections from the overlay.
    #[test]
    fn vehicle_draw_spec_contains_all_labels() -> anyhow::Result<()> {
        let spec = build_vehicle_draw_spec()?;
        for label in ["car", "motorbike", "bus", "truck"] {
            assert!(
                spec.lookup(DRAW_NAMESPACE, label).is_some(),
                "missing draw spec for {label}"
            );
        }
        Ok(())
    }
}
