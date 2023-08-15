use crate::consts::EPS;
use crate::draw::PaddingDraw;
use crate::primitives::object::VideoObject;
use crate::primitives::{Point, PolygonalArea};
use crate::round_2_digits;
use crate::to_json_value::ToSerdeJsonValue;
use anyhow::{bail, Result};
use geo::{Area, BooleanOps};
use parking_lot::RwLock;
use rkyv::{Archive, Deserialize, Serialize};
use serde_json::Value;
use std::f32::consts::PI;
use std::sync::Arc;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename = "bbox.metric.type")]
pub enum BBoxMetricType {
    IoU,
    IoSelf,
    IoOther,
}

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[archive(check_bytes)]
pub struct OwnedRBBoxData {
    pub xc: f32,
    pub yc: f32,
    pub width: f32,
    pub height: f32,
    pub angle: Option<f32>,
    pub has_modifications: bool,
}

impl Default for OwnedRBBoxData {
    fn default() -> Self {
        Self {
            xc: 0.0,
            yc: 0.0,
            width: 0.0,
            height: 0.0,
            angle: None,
            has_modifications: false,
        }
    }
}

impl ToSerdeJsonValue for OwnedRBBoxData {
    fn to_serde_json_value(&self) -> Value {
        serde_json::json!({
            "xc": self.xc,
            "yc": self.yc,
            "width": self.width,
            "height": self.height,
            "angle": self.angle,
        })
    }
}

impl TryFrom<RBBox> for OwnedRBBoxData {
    type Error = anyhow::Error;

    fn try_from(value: RBBox) -> Result<Self, Self::Error> {
        OwnedRBBoxData::try_from(&value)
    }
}

impl TryFrom<&RBBox> for OwnedRBBoxData {
    type Error = anyhow::Error;

    fn try_from(value: &RBBox) -> Result<Self, Self::Error> {
        match &value.data {
            BBoxVariant::Owned(d) => Ok(d.clone()),
            BBoxVariant::BorrowedDetectionBox(d) => Ok(d.read().detection_box.clone()),
            BBoxVariant::BorrowedTrackingBox(d) => d.read().track_box.as_ref().map_or_else(
                || Err(anyhow::anyhow!("Cannot convert tracking box to RBBoxData")),
                |t| Ok(t.clone()),
            ),
        }
    }
}

#[derive(Debug, Clone)]
enum BBoxVariant {
    Owned(OwnedRBBoxData),
    BorrowedDetectionBox(Arc<RwLock<VideoObject>>),
    BorrowedTrackingBox(Arc<RwLock<VideoObject>>),
}

/// Represents a bounding box with an optional rotation angle in degrees.
///
#[derive(Debug, Clone)]
pub struct RBBox {
    data: BBoxVariant,
}

impl PartialEq for RBBox {
    fn eq(&self, other: &Self) -> bool {
        self.geometric_eq(other)
    }
}

impl ToSerdeJsonValue for RBBox {
    fn to_serde_json_value(&self) -> Value {
        serde_json::json!({
            "xc": self.get_xc(),
            "yc": self.get_yc(),
            "width": self.get_width(),
            "height": self.get_height(),
            "angle": self.get_angle(),
        })
    }
}

impl From<OwnedRBBoxData> for RBBox {
    fn from(value: OwnedRBBoxData) -> Self {
        Self {
            data: BBoxVariant::Owned(value),
        }
    }
}

impl RBBox {
    pub fn borrowed_detection_box(object: Arc<RwLock<VideoObject>>) -> Self {
        Self {
            data: BBoxVariant::BorrowedDetectionBox(object),
        }
    }

    pub fn borrowed_track_box(object: Arc<RwLock<VideoObject>>) -> Self {
        Self {
            data: BBoxVariant::BorrowedTrackingBox(object),
        }
    }
}

impl RBBox {
    pub fn get_area(&self) -> f32 {
        self.get_width() * self.get_height()
    }

    pub fn geometric_eq(&self, other: &Self) -> bool {
        self.get_xc() == other.get_xc()
            && self.get_yc() == other.get_yc()
            && self.get_width() == other.get_width()
            && self.get_height() == other.get_height()
            && self.get_angle() == other.get_angle()
    }
    pub fn almost_eq(&self, other: &Self, eps: f32) -> bool {
        (self.get_xc() - other.get_xc()).abs() < eps
            && (self.get_yc() - other.get_yc()).abs() < eps
            && (self.get_width() - other.get_width()).abs() < eps
            && (self.get_height() - other.get_height()).abs() < eps
            && (self.get_angle().unwrap_or(0.0) - other.get_angle().unwrap_or(0.0)).abs() < eps
    }
    pub fn get_xc(&self) -> f32 {
        match &self.data {
            BBoxVariant::Owned(d) => d.xc,
            BBoxVariant::BorrowedDetectionBox(d) => d.read_recursive().detection_box.xc,
            BBoxVariant::BorrowedTrackingBox(d) => d
                .read_recursive()
                .track_box
                .as_ref()
                .map(|t| t.xc)
                .unwrap_or(0.0),
        }
    }
    pub fn get_yc(&self) -> f32 {
        match &self.data {
            BBoxVariant::Owned(d) => d.yc,
            BBoxVariant::BorrowedDetectionBox(d) => d.read_recursive().detection_box.yc,
            BBoxVariant::BorrowedTrackingBox(d) => d
                .read_recursive()
                .track_box
                .as_ref()
                .map(|t| t.yc)
                .unwrap_or(0.0),
        }
    }
    pub fn get_width(&self) -> f32 {
        match &self.data {
            BBoxVariant::Owned(d) => d.width,
            BBoxVariant::BorrowedDetectionBox(d) => d.read_recursive().detection_box.width,
            BBoxVariant::BorrowedTrackingBox(d) => d
                .read_recursive()
                .track_box
                .as_ref()
                .map(|t| t.width)
                .unwrap_or(0.0),
        }
    }
    pub fn get_height(&self) -> f32 {
        match &self.data {
            BBoxVariant::Owned(d) => d.height,
            BBoxVariant::BorrowedDetectionBox(d) => d.read_recursive().detection_box.height,
            BBoxVariant::BorrowedTrackingBox(d) => d
                .read_recursive()
                .track_box
                .as_ref()
                .map(|t| t.height)
                .unwrap_or(0.0),
        }
    }
    pub fn get_angle(&self) -> Option<f32> {
        match &self.data {
            BBoxVariant::Owned(d) => d.angle,
            BBoxVariant::BorrowedDetectionBox(d) => d.read_recursive().detection_box.angle,
            BBoxVariant::BorrowedTrackingBox(d) => d
                .read_recursive()
                .track_box
                .as_ref()
                .map(|t| t.angle)
                .unwrap_or(None),
        }
    }
    pub fn get_width_to_height_ratio(&self) -> f32 {
        let height = self.get_height();
        if height == 0.0 {
            // TODO: should we return an error here?
            return -1.0;
        }
        self.get_width() / self.get_height()
    }
    pub fn is_modified(&self) -> bool {
        match &self.data {
            BBoxVariant::Owned(d) => d.has_modifications,
            BBoxVariant::BorrowedDetectionBox(d) => {
                d.read_recursive().detection_box.has_modifications
            }
            BBoxVariant::BorrowedTrackingBox(d) => d
                .read_recursive()
                .track_box
                .as_ref()
                .map(|t| t.has_modifications)
                .unwrap_or(false),
        }
    }
    pub fn set_modifications(&mut self, value: bool) {
        match &mut self.data {
            BBoxVariant::Owned(d) => d.has_modifications = value,
            BBoxVariant::BorrowedDetectionBox(d) => {
                let mut lock = d.write();
                lock.detection_box.has_modifications = value;
            }
            BBoxVariant::BorrowedTrackingBox(d) => {
                let mut lock = d.write();
                if let Some(b) = &mut lock.track_box {
                    b.has_modifications = value;
                }
            }
        }
    }
    pub fn set_xc(&mut self, xc: f32) {
        match &mut self.data {
            BBoxVariant::Owned(d) => {
                d.xc = xc;
                d.has_modifications = true;
            }
            BBoxVariant::BorrowedDetectionBox(d) => {
                let mut lock = d.write();
                lock.detection_box.xc = xc;
                lock.detection_box.has_modifications = true;
            }
            BBoxVariant::BorrowedTrackingBox(d) => {
                let mut lock = d.write();
                if let Some(b) = &mut lock.track_box {
                    b.xc = xc;
                    b.has_modifications = true;
                }
            }
        }
    }
    pub fn set_yc(&mut self, yc: f32) {
        match &mut self.data {
            BBoxVariant::Owned(d) => {
                d.yc = yc;
                d.has_modifications = true;
            }
            BBoxVariant::BorrowedDetectionBox(d) => {
                let mut lock = d.write();
                lock.detection_box.yc = yc;
                lock.detection_box.has_modifications = true;
            }
            BBoxVariant::BorrowedTrackingBox(d) => {
                let mut lock = d.write();
                if let Some(b) = &mut lock.track_box {
                    b.yc = yc;
                    b.has_modifications = true;
                }
            }
        }
    }
    pub fn set_width(&mut self, width: f32) {
        match &mut self.data {
            BBoxVariant::Owned(d) => {
                d.width = width;
                d.has_modifications = true;
            }
            BBoxVariant::BorrowedDetectionBox(d) => {
                let mut lock = d.write();
                lock.detection_box.width = width;
                lock.detection_box.has_modifications = true;
            }
            BBoxVariant::BorrowedTrackingBox(d) => {
                let mut lock = d.write();
                if let Some(b) = &mut lock.track_box {
                    b.width = width;
                    b.has_modifications = true;
                }
            }
        }
    }
    pub fn set_height(&mut self, height: f32) {
        match &mut self.data {
            BBoxVariant::Owned(d) => {
                d.height = height;
                d.has_modifications = true;
            }
            BBoxVariant::BorrowedDetectionBox(d) => {
                let mut lock = d.write();
                lock.detection_box.height = height;
                lock.detection_box.has_modifications = true;
            }
            BBoxVariant::BorrowedTrackingBox(d) => {
                let mut lock = d.write();
                if let Some(b) = &mut lock.track_box {
                    b.height = height;
                    b.has_modifications = true;
                }
            }
        }
    }
    pub fn set_angle(&mut self, angle: Option<f32>) {
        match &mut self.data {
            BBoxVariant::Owned(d) => {
                d.angle = angle;
                d.has_modifications = true;
            }
            BBoxVariant::BorrowedDetectionBox(d) => {
                let mut lock = d.write();
                lock.detection_box.angle = angle;
                lock.detection_box.has_modifications = true;
            }
            BBoxVariant::BorrowedTrackingBox(d) => {
                let mut lock = d.write();
                if let Some(b) = &mut lock.track_box {
                    b.angle = angle;
                    b.has_modifications = true;
                }
            }
        }
    }
    pub fn new(xc: f32, yc: f32, width: f32, height: f32, angle: Option<f32>) -> Self {
        Self {
            data: BBoxVariant::Owned(OwnedRBBoxData {
                xc,
                yc,
                width,
                height,
                angle,
                has_modifications: false,
            }),
        }
    }
    pub fn shift(&mut self, dx: f32, dy: f32) {
        match &mut self.data {
            BBoxVariant::Owned(d) => {
                d.xc += dx;
                d.yc += dy;
                d.has_modifications = true;
            }
            BBoxVariant::BorrowedDetectionBox(d) => {
                let mut lock = d.write();
                lock.detection_box.xc += dx;
                lock.detection_box.yc += dy;
                lock.detection_box.has_modifications = true;
            }
            BBoxVariant::BorrowedTrackingBox(d) => {
                let mut lock = d.write();
                if let Some(track_box) = &mut lock.track_box {
                    track_box.xc += dx;
                    track_box.yc += dy;
                    track_box.has_modifications = true;
                }
            }
        }
    }

    pub fn ltrb(left: f32, top: f32, right: f32, bottom: f32) -> Self {
        let width = right - left;
        let height = bottom - top;

        let xc = (left + right) / 2.0;
        let yc = (top + bottom) / 2.0;

        Self::new(xc, yc, width, height, None)
    }

    pub fn ltwh(left: f32, top: f32, width: f32, height: f32) -> Self {
        let xc = left + width / 2.0;
        let yc = top + height / 2.0;
        RBBox::new(xc, yc, width, height, None)
    }
    pub fn get_top(&self) -> Result<f32> {
        if self.get_angle().unwrap_or(0.0) == 0.0 {
            Ok(self.get_yc() - self.get_height() / 2.0)
        } else {
            bail!("Cannot get top for rotated bounding box",);
        }
    }
    pub fn set_top(&mut self, top: f32) -> anyhow::Result<()> {
        if self.get_angle().unwrap_or(0.0) == 0.0 {
            self.set_modifications(true);
            let h = self.get_height();
            self.set_yc(top + h / 2.0);
            Ok(())
        } else {
            bail!("Cannot set top for rotated bounding box",)
        }
    }
    pub fn get_left(&self) -> Result<f32> {
        if self.get_angle().unwrap_or(0.0) == 0.0 {
            Ok(self.get_xc() - self.get_width() / 2.0)
        } else {
            bail!("Cannot get left for rotated bounding box",)
        }
    }

    pub fn set_left(&mut self, left: f32) -> Result<()> {
        if self.get_angle().unwrap_or(0.0) == 0.0 {
            self.set_modifications(true);
            let w = self.get_width();
            self.set_xc(left + w / 2.0);
            Ok(())
        } else {
            bail!("Cannot set left for rotated bounding box",)
        }
    }

    pub fn get_right(&self) -> Result<f32> {
        if self.get_angle().unwrap_or(0.0) == 0.0 {
            Ok(self.get_xc() + self.get_width() / 2.0)
        } else {
            bail!("Cannot get right for rotated bounding box",)
        }
    }

    pub fn get_bottom(&self) -> Result<f32> {
        if self.get_angle().unwrap_or(0.0) == 0.0 {
            Ok(self.get_yc() + self.get_height() / 2.0)
        } else {
            bail!("Cannot get bottom for rotated bounding box",)
        }
    }

    /// Returns (left, top, right, bottom) coordinates.
    ///
    pub fn as_ltrb(&self) -> Result<(f32, f32, f32, f32)> {
        if self.get_angle().unwrap_or(0.0) != 0.0 {
            bail!("Cannot get left, top, width, height for rotated bounding box",)
        }
        let top = self.get_top()?;
        let left = self.get_left()?;
        let bottom = self.get_bottom()?;
        let right = self.get_right()?;

        Ok((left, top, right, bottom))
    }

    /// Returns (left, top, right, bottom) coordinates rounded to integers.
    ///
    pub fn as_ltrb_int(&self) -> Result<(i64, i64, i64, i64)> {
        if self.get_angle().unwrap_or(0.0) != 0.0 {
            bail!("Cannot get left, top, width, height for rotated bounding box",)
        }
        let top = self.get_top()?.floor();
        let left = self.get_left()?.floor();
        let bottom = self.get_bottom()?.ceil();
        let right = self.get_right()?.ceil();

        Ok((left as i64, top as i64, right as i64, bottom as i64))
    }

    /// Returns (left, top, width, height) coordinates.
    ///
    pub fn as_ltwh(&self) -> Result<(f32, f32, f32, f32)> {
        if self.get_angle().unwrap_or(0.0) != 0.0 {
            bail!("Cannot get left, top, width, height for rotated bounding box",)
        }
        let top = self.get_top()?;
        let left = self.get_left()?;
        let width = self.get_width();
        let height = self.get_height();
        Ok((left, top, width, height))
    }

    /// Returns (left, top, width, height) coordinates rounded to integers.
    ///
    pub fn as_ltwh_int(&self) -> Result<(i64, i64, i64, i64)> {
        if self.get_angle().unwrap_or(0.0) != 0.0 {
            bail!("Cannot get left, top, width, height for rotated bounding box",)
        }
        let top = self.get_top()?.floor();
        let left = self.get_left()?.floor();
        let width = self.get_width().ceil();
        let height = self.get_height().ceil();
        Ok((left as i64, top as i64, width as i64, height as i64))
    }

    /// Returns (xc, yc, width, height) coordinates.
    ///
    pub fn as_xcycwh(&self) -> (f32, f32, f32, f32) {
        let xc = self.get_xc();
        let yc = self.get_yc();
        let width = self.get_width();
        let height = self.get_height();
        (xc, yc, width, height)
    }

    /// Returns (xc, yc, width, height) coordinates rounded to integers.
    ///
    pub fn as_xcycwh_int(&self) -> (i64, i64, i64, i64) {
        let xc = self.get_xc();
        let yc = self.get_yc();
        let width = self.get_width();
        let height = self.get_height();
        (xc as i64, yc as i64, width as i64, height as i64)
    }
}

impl RBBox {
    pub(crate) fn intersection_coaxial(&self, other: &RBBox) -> Option<f32> {
        if self.get_angle().unwrap_or(0.0) != other.get_angle().unwrap_or(0.0) {
            return None;
        }
        let mut bb1 = self.clone();
        bb1.set_angle(None);
        let mut bb2 = other.clone();
        bb2.set_angle(None);

        let (xmin1, ymin1, xmax1, ymax1) = bb1.as_ltrb().unwrap();
        let (xmin2, ymin2, xmax2, ymax2) = bb2.as_ltrb().unwrap();

        // Calculate the overlap coordinates
        let overlap_xmin = xmin1.max(xmin2);
        let overlap_ymin = ymin1.max(ymin2);
        let overlap_xmax = xmax1.min(xmax2);
        let overlap_ymax = ymax1.min(ymax2);

        // Calculate the overlap area
        let intersection = if overlap_xmin < overlap_xmax && overlap_ymin < overlap_ymax {
            (overlap_xmax - overlap_xmin) * (overlap_ymax - overlap_ymin)
        } else {
            0.0
        };

        Some(intersection)
    }

    pub fn new_padded(&self, padding: &PaddingDraw) -> Self {
        let (left, right, top, bottom) = (
            padding.left as f32,
            padding.right as f32,
            padding.top as f32,
            padding.bottom as f32,
        );

        let xc = self.get_xc();
        let yc = self.get_yc();
        let width = self.get_width();
        let height = self.get_height();
        let angle = self.get_angle();

        let angle_rad = angle.unwrap_or(0.0) * PI / 180.0;
        let cos_theta = angle_rad.cos();
        let sin_theta = angle_rad.sin();

        let xc = xc + ((right - left) * cos_theta - (bottom - top) * sin_theta) / 2.0;
        let yc = yc + ((right - left) * sin_theta + (bottom - top) * cos_theta) / 2.0;
        let height = height + top + bottom;
        let width = width + left + right;

        Self::new(xc, yc, width, height, angle)
    }

    fn calculate_intersection(&self, other: &Self) -> Result<f32> {
        if self.get_area() < EPS || other.get_area() < EPS {
            bail!("Area of one of the bounding boxes is zero. Division by zero is not allowed.");
        }

        Ok(if let Some(int) = self.intersection_coaxial(other) {
            int
        } else {
            let mut area1 = self.get_as_polygonal_area();
            let poly1 = area1.get_polygon();
            let mut area2 = other.get_as_polygonal_area();
            let poly2 = area2.get_polygon();
            poly1.intersection(&poly2).unsigned_area() as f32
        })
    }

    pub fn ios(&self, other: &Self) -> Result<f32> {
        let own_area = self.get_area();
        Ok(self.calculate_intersection(other)? / own_area)
    }

    pub fn ioo(&self, other: &Self) -> Result<f32> {
        let other_area = other.get_area();
        Ok(self.calculate_intersection(other)? / other_area)
    }

    pub fn iou(&self, other: &Self) -> Result<f32> {
        let intersection = self.calculate_intersection(other)?;
        Ok(intersection / (self.get_area() + other.get_area() - intersection))
    }

    pub fn scale(&mut self, scale_x: f32, scale_y: f32) {
        let angle = self.get_angle().unwrap_or(0.0);
        let xc = self.get_xc();
        let yc = self.get_yc();
        let width = self.get_width();
        let height = self.get_height();

        if angle % 90.0 == 0.0 {
            self.set_xc(xc * scale_x);
            self.set_yc(yc * scale_y);
            self.set_width(width * scale_x);
            self.set_height(height * scale_y);
        } else {
            let scale_x2 = scale_x * scale_x;
            let scale_y2 = scale_y * scale_y;
            let cotan = (angle * PI / 180.0).tan().powi(-1);
            let cotan_2 = cotan * cotan;
            let scale_angle =
                (scale_x * angle.signum() / (scale_x2 + scale_y2 * cotan_2).sqrt()).acos();
            let nscale_height = ((scale_x2 + scale_y2 * cotan_2) / (1.0 + cotan_2)).sqrt();
            let ayh = 1.0 / ((90.0 - angle) / 180.0 * PI).tan();
            let nscale_width = ((scale_x2 + scale_y2 * ayh * ayh) / (1.0 + ayh * ayh)).sqrt();

            self.set_angle(Some(90.0 - (scale_angle * 180.0 / PI)));
            self.set_xc(xc * scale_x);
            self.set_yc(yc * scale_y);
            self.set_width(width * nscale_width);
            self.set_height(height * nscale_height);
        }
    }

    pub fn get_vertices(&self) -> Vec<(f32, f32)> {
        let angle = self.get_angle().unwrap_or(0.0);

        let x = self.get_xc();
        let y = self.get_yc();
        let w = self.get_width() / 2.0;
        let h = self.get_height() / 2.0;

        if angle == 0.0 {
            vec![
                (x - w, y - h),
                (x + w, y - h),
                (x + w, y + h),
                (x - w, y + h),
            ]
        } else {
            let angle = angle * PI / 180.0;
            let cos = angle.cos();
            let sin = angle.sin();
            vec![
                (x + w * cos - h * sin, y + w * sin + h * cos),
                (x + w * cos + h * sin, y + w * sin - h * cos),
                (x - w * cos + h * sin, y - w * sin - h * cos),
                (x - w * cos - h * sin, y - w * sin + h * cos),
            ]
        }
    }

    pub fn get_vertices_rounded(&self) -> Vec<(f32, f32)> {
        self.get_vertices()
            .into_iter()
            .map(|(x, y)| (round_2_digits(x), round_2_digits(y)))
            .collect::<Vec<_>>()
    }

    pub fn get_vertices_int(&self) -> Vec<(i64, i64)> {
        self.get_vertices()
            .into_iter()
            .map(|(x, y)| (x as i64, y as i64))
            .collect::<Vec<_>>()
    }

    pub fn get_as_polygonal_area(&self) -> PolygonalArea {
        PolygonalArea::new(
            self.get_vertices()
                .into_iter()
                .map(|(x, y)| Point::new(x, y))
                .collect::<Vec<_>>(),
            None,
        )
    }

    pub fn get_wrapping_bbox(&self) -> RBBox {
        if self.get_angle().is_none() {
            RBBox::new(
                self.get_xc(),
                self.get_yc(),
                self.get_width(),
                self.get_height(),
                None,
            )
        } else {
            let mut vertices = self.get_vertices();
            let (initial_vtx_x, initial_vtx_y) = vertices.pop().unwrap();
            let (mut min_x, mut min_y, mut max_x, mut max_y) =
                (initial_vtx_x, initial_vtx_y, initial_vtx_x, initial_vtx_y);
            for v in vertices {
                let (vtx_x, vtx_y) = v;
                if vtx_x < min_x {
                    min_x = vtx_x;
                }
                if vtx_x > max_x {
                    max_x = vtx_x;
                }
                if vtx_y < min_y {
                    min_y = vtx_y;
                }
                if vtx_y > max_y {
                    max_y = vtx_y;
                }
            }
            RBBox::new(
                (min_x + max_x) / 2.0,
                (min_y + max_y) / 2.0,
                max_x - min_x,
                max_y - min_y,
                None,
            )
        }
    }

    pub fn get_visual_bbox(&self, padding: &PaddingDraw, border_width: i64) -> Result<RBBox> {
        if border_width < 0 {
            bail!("border_width must be greater than or equal to 0",)
        }
        let padding_with_border = PaddingDraw::new(
            padding.left + border_width,
            padding.top + border_width,
            padding.right + border_width,
            padding.bottom + border_width,
        )?;

        Ok(self.new_padded(&padding_with_border))
    }
}

#[cfg(test)]
mod tests {
    use crate::draw::PaddingDraw;
    use crate::primitives::RBBox;
    use crate::round_2_digits;

    #[test]
    fn test_scale_no_angle() {
        let mut bbox = RBBox::new(0.0, 0.0, 100.0, 100.0, None);
        bbox.scale(2.0, 2.0);
        assert_eq!(bbox.get_xc(), 0.0);
        assert_eq!(bbox.get_yc(), 0.0);
        assert_eq!(bbox.get_width(), 200.0);
        assert_eq!(bbox.get_height(), 200.0);
        assert_eq!(bbox.get_angle(), None);
    }

    #[test]
    fn test_scale_with_angle() {
        let mut bbox = RBBox::new(0.0, 0.0, 100.0, 100.0, Some(45.0));
        bbox.scale(2.0, 3.0);
        //dbg!(&bbox);
        assert_eq!(bbox.get_xc(), 0.0);
        assert_eq!(bbox.get_yc(), 0.0);
        assert_eq!(round_2_digits(bbox.get_width()), 254.95);
        assert_eq!(round_2_digits(bbox.get_height()), 254.95);
        assert_eq!(bbox.get_angle().map(round_2_digits), Some(33.69));
    }

    #[test]
    fn test_vertices() {
        let bbox = RBBox::new(0.0, 0.0, 100.0, 100.0, Some(45.0));
        let vertices = bbox.get_vertices_rounded();
        assert_eq!(vertices.len(), 4);
        assert_eq!(vertices[0], (0.0, 70.71));
        assert_eq!(vertices[1], (70.71, 0.0));
        assert_eq!(vertices[2], (-0.0, -70.71));
        assert_eq!(vertices[3], (-70.71, 0.0));
    }

    #[test]
    fn test_wrapping_bbox() {
        let bbox = RBBox::new(0.0, 0.0, 100.0, 100.0, Some(45.0));
        let wrapped = bbox.get_wrapping_bbox();
        assert_eq!(wrapped.get_xc(), 0.0);
        assert_eq!(wrapped.get_yc(), 0.0);
        assert_eq!(round_2_digits(wrapped.get_width()), 141.42);
        assert_eq!(round_2_digits(wrapped.get_height()), 141.42);
        assert_eq!(wrapped.get_angle(), None);

        let bbox = RBBox::new(0.0, 0.0, 50.0, 100.0, None);
        let wrapped = bbox.get_wrapping_bbox();
        assert_eq!(wrapped.get_xc(), 0.0);
        assert_eq!(wrapped.get_yc(), 0.0);
        assert_eq!(round_2_digits(wrapped.get_width()), 50.0);
        assert_eq!(round_2_digits(wrapped.get_height()), 100.0);
        assert_eq!(wrapped.get_angle(), None);

        let bbox = RBBox::new(0.0, 0.0, 50.0, 100.0, Some(90.0));
        let wrapped = bbox.get_wrapping_bbox();
        assert_eq!(wrapped.get_xc(), 0.0);
        assert_eq!(wrapped.get_yc(), 0.0);
        assert_eq!(round_2_digits(wrapped.get_width()), 100.0);
        assert_eq!(round_2_digits(wrapped.get_height()), 50.0);
        assert_eq!(wrapped.get_angle(), None);
    }

    fn get_bbox(angle: Option<f32>) -> RBBox {
        RBBox::new(0.0, 0.0, 100.0, 100.0, angle)
    }

    #[test]
    fn check_modifications() {
        let mut bb = get_bbox(Some(45.0));
        bb.set_xc(10.0);
        assert!(bb.is_modified());

        let mut bb = get_bbox(Some(45.0));
        bb.set_yc(10.0);
        assert!(bb.is_modified());

        let mut bb = get_bbox(Some(45.0));
        bb.set_width(10.0);
        assert!(bb.is_modified());

        let mut bb = get_bbox(Some(45.0));
        bb.set_height(10.0);
        assert!(bb.is_modified());

        let mut bb = get_bbox(Some(45.0));
        bb.set_angle(Some(10.0));
        assert!(bb.is_modified());

        let mut bb = get_bbox(Some(45.0));
        bb.set_angle(None);
        assert!(bb.is_modified());

        let mut bb = get_bbox(Some(45.0));
        bb.scale(2.0, 2.0);
        assert!(bb.is_modified());
    }

    #[test]
    fn test_padded_axis_aligned() {
        let bb = get_bbox(None);
        let padded = bb.new_padded(&PaddingDraw::new(0, 0, 0, 0).unwrap());
        assert_eq!(padded.get_xc(), bb.get_xc());
        assert_eq!(padded.get_yc(), bb.get_yc());
        assert_eq!(padded.get_width(), bb.get_width());
        assert_eq!(padded.get_height(), bb.get_height());

        let bb = get_bbox(None);
        let padded = bb.new_padded(&PaddingDraw::new(2, 0, 0, 0).unwrap());
        assert_eq!(padded.get_xc(), bb.get_xc() - 1.0);
        assert_eq!(padded.get_yc(), bb.get_yc());
        assert_eq!(padded.get_width(), bb.get_width() + 2.0);
        assert_eq!(padded.get_height(), bb.get_height());

        let bb = get_bbox(None);
        let padded = bb.new_padded(&PaddingDraw::new(0, 2, 0, 0).unwrap());
        assert_eq!(padded.get_xc(), bb.get_xc());
        assert_eq!(padded.get_yc(), bb.get_yc() - 1.0);
        assert_eq!(padded.get_width(), bb.get_width());
        assert_eq!(padded.get_height(), bb.get_height() + 2.0);

        let bb = get_bbox(None);
        let padded = bb.new_padded(&PaddingDraw::new(2, 0, 4, 0).unwrap());
        assert_eq!(padded.get_xc(), bb.get_xc() + 1.0);
        assert_eq!(padded.get_yc(), bb.get_yc());
        assert_eq!(padded.get_width(), bb.get_width() + 6.0);
        assert_eq!(padded.get_height(), bb.get_height());
    }

    #[test]
    fn test_padded_rotated() {
        let bb = get_bbox(Some(90.0));
        let padded = bb.new_padded(&PaddingDraw::new(2, 0, 0, 0).unwrap());
        assert_eq!(round_2_digits(padded.get_xc()), bb.get_xc());
        assert_eq!(round_2_digits(padded.get_yc()), bb.get_yc() - 1.0);
        assert_eq!(padded.get_width(), bb.get_width() + 2.0);
        assert_eq!(padded.get_height(), bb.get_height());
    }

    #[test]
    fn test_eq() {
        let bb1 = get_bbox(Some(45.0));
        let bb2 = get_bbox(Some(45.0));
        assert_eq!(bb1, bb2);

        let bb1 = get_bbox(Some(45.0));
        let bb2 = get_bbox(Some(90.0));
        assert_ne!(bb1, bb2);

        let bb1 = get_bbox(Some(45.0));
        let bb2 = get_bbox(None);
        assert_ne!(bb1, bb2);
    }

    #[test]
    fn test_almost_eq() {
        let bb1 = get_bbox(Some(45.0));
        let bb2 = get_bbox(Some(45.05));
        assert!(bb1.almost_eq(&bb2, 0.1));

        let bb1 = get_bbox(Some(45.0));
        let bb2 = get_bbox(Some(90.0));
        assert!(!bb1.almost_eq(&bb2, 0.1));

        let bb1 = get_bbox(Some(45.0));
        let bb2 = get_bbox(None);
        assert!(!bb1.almost_eq(&bb2, 0.1));
    }

    #[test]
    fn test_shift() {
        let mut bb = get_bbox(Some(45.0));
        bb.shift(10.0, 20.0);
        assert_eq!(bb.get_xc(), 10.0);
        assert_eq!(bb.get_yc(), 20.0);
    }

    #[test]
    fn test_various_reprs_non_zero_angle() {
        let mut bb = get_bbox(Some(45.0));
        assert!(bb.as_ltrb().is_err());
        assert!(bb.as_ltrb_int().is_err());
        assert!(bb.as_ltwh().is_err());
        assert!(bb.as_ltwh_int().is_err());
        assert!(bb.get_top().is_err());
        assert!(bb.get_left().is_err());
        assert!(bb.get_bottom().is_err());
        assert!(bb.get_right().is_err());
        assert!(bb.set_top(11.0).is_err());
        assert!(bb.set_left(12.0).is_err());
        assert!(!bb.is_modified());
    }

    #[test]
    fn test_various_reprs_zero_angle() {
        let mut bb = get_bbox(Some(0.0));
        assert!(bb.as_ltrb().is_ok());
        assert!(bb.as_ltrb_int().is_ok());
        assert!(bb.as_ltwh().is_ok());
        assert!(bb.as_ltwh_int().is_ok());
        assert!(bb.get_top().is_ok());
        assert!(bb.get_left().is_ok());
        assert!(bb.get_bottom().is_ok());
        assert!(bb.get_right().is_ok());
        assert!(bb.set_top(11.0).is_ok());
        assert!(bb.set_left(12.0).is_ok());
        assert!(bb.is_modified());
    }

    #[test]
    fn test_various_reprs_none_angle() {
        let mut bb = get_bbox(None);
        assert!(bb.as_ltrb().is_ok());
        assert!(bb.as_ltrb_int().is_ok());
        assert!(bb.as_ltwh().is_ok());
        assert!(bb.as_ltwh_int().is_ok());
        assert!(bb.get_top().is_ok());
        assert!(bb.get_left().is_ok());
        assert!(bb.get_bottom().is_ok());
        assert!(bb.get_right().is_ok());
        assert!(bb.set_top(11.0).is_ok());
        assert!(bb.set_left(12.0).is_ok());
        assert!(bb.is_modified());
    }

    #[test]
    fn test_reprs_correct_values() {
        let mut bb = get_bbox(None);
        bb.set_xc(10.0);
        bb.set_yc(20.0);
        bb.set_width(30.0);
        bb.set_height(40.0);
        let left = bb.get_left().unwrap();
        assert_eq!(left, 10.0 - 30.0 / 2.0);
        let top = bb.get_top().unwrap();
        assert_eq!(top, 20.0 - 40.0 / 2.0);
        let right = bb.get_right().unwrap();
        assert_eq!(right, 10.0 + 30.0 / 2.0);
        let bottom = bb.get_bottom().unwrap();
        assert_eq!(bottom, 20.0 + 40.0 / 2.0);
        let width = bb.get_width();
        let height = bb.get_height();
        let ltrb = bb.as_ltrb().unwrap();
        assert_eq!(ltrb, (left, top, right, bottom));
        let ltrb_int = bb.as_ltrb_int().unwrap();
        assert_eq!(
            ltrb_int,
            (
                left.floor() as i64,
                top.floor() as i64,
                right.ceil() as i64,
                bottom.ceil() as i64
            )
        );

        let ltwh = bb.as_ltwh().unwrap();
        assert_eq!(ltwh, (left, top, width, height));
        let ltwh_int = bb.as_ltwh_int().unwrap();
        assert_eq!(
            ltwh_int,
            (
                left.floor() as i64,
                top.floor() as i64,
                width.ceil() as i64,
                height.ceil() as i64
            )
        );
    }

    #[test]
    fn test_intersection_coaxial() {
        let bb1 = RBBox::new(0.0, 0.0, 100.0, 100.0, Some(0.0));
        let bb2 = RBBox::new(0.0, 0.0, 100.0, 100.0, Some(0.0));
        let int = bb1.intersection_coaxial(&bb2).unwrap();
        assert_eq!(int, 10000.0);

        let bb1 = RBBox::new(0.0, 0.0, 100.0, 100.0, Some(20.0));
        let bb2 = RBBox::new(0.0, 0.0, 100.0, 100.0, Some(20.0));
        let int = bb1.intersection_coaxial(&bb2).unwrap();
        assert_eq!(int, 10000.0);

        let bb1 = RBBox::new(0.0, 0.0, 100.0, 100.0, None);
        let bb2 = RBBox::new(0.0, 0.0, 100.0, 100.0, None);
        let int = bb1.intersection_coaxial(&bb2).unwrap();
        assert_eq!(int, 10000.0);

        let bb1 = RBBox::new(0.0, 0.0, 100.0, 100.0, Some(1.0));
        let bb2 = RBBox::new(0.0, 0.0, 100.0, 100.0, None);
        let int = bb1.intersection_coaxial(&bb2);
        assert!(int.is_none());

        let int = bb2.intersection_coaxial(&bb1);
        assert!(int.is_none());

        let bb1 = RBBox::new(0.0, 0.0, 100.0, 100.0, None);
        let bb2 = RBBox::new(0.0, 0.0, 50.0, 50.0, None);

        let int = bb1.intersection_coaxial(&bb2).unwrap();
        assert_eq!(int, 2500.0);

        let int = bb2.intersection_coaxial(&bb1).unwrap();
        assert_eq!(int, 2500.0);

        let bb1 = RBBox::new(-20.0, 0.0, 40.0, 100.0, Some(0.0));
        let bb2 = RBBox::new(0.0, 0.0, 40.0, 100.0, Some(0.0));
        let int = bb1.intersection_coaxial(&bb2).unwrap();
        assert_eq!(int, 2000.0);

        let int = bb2.intersection_coaxial(&bb1).unwrap();
        assert_eq!(int, 2000.0);
    }
}
