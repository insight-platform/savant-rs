//! Safe wrapper for DeepStream rectangle parameters

use crate::{DeepStreamError, Result};
use deepstream_sys::NvOSD_RectParams;

/// Safe wrapper for DeepStream rectangle parameters
///
/// This struct provides safe access to rectangle parameters while managing
/// the underlying C memory properly.
#[derive(Debug, Clone)]
pub struct RectParams {
    /// Raw pointer to the C structure
    raw: *mut NvOSD_RectParams,
    /// Whether this instance owns the memory
    owned: bool,
}

impl RectParams {
    /// Create a new rectangle parameters instance
    pub fn new(left: f32, top: f32, width: f32, height: f32) -> Self {
        // Create a new NvOSD_RectParams structure
        let rect = NvOSD_RectParams {
            left,
            top,
            width,
            height,
            border_width: 0,
            border_color: deepstream_sys::NvOSD_ColorParams {
                red: 0.0,
                green: 0.0,
                blue: 0.0,
                alpha: 0.0,
            },
            has_bg_color: 0u32,
            reserved: 0u32,
            bg_color: deepstream_sys::NvOSD_ColorParams {
                red: 0.0,
                green: 0.0,
                blue: 0.0,
                alpha: 0.0,
            },
            has_color_info: 0i32,
            color_id: 0i32,
        };

        // Allocate memory for the structure
        let raw = Box::into_raw(Box::new(rect));

        Self { raw, owned: true }
    }

    /// Create from a raw pointer
    ///
    /// # Safety
    /// The caller must ensure the pointer is valid and not null.
    pub unsafe fn from_raw(raw: *mut NvOSD_RectParams) -> Result<Self> {
        if raw.is_null() {
            return Err(DeepStreamError::null_pointer("RectParams::from_raw"));
        }

        Ok(Self { raw, owned: false })
    }

    /// Create from a reference to raw structure
    ///
    /// # Safety
    /// The caller must ensure the reference is valid.
    pub unsafe fn from_ref(raw: &NvOSD_RectParams) -> Self {
        Self {
            raw: raw as *const _ as *mut _,
            owned: false,
        }
    }

    /// Get the raw pointer
    ///
    /// # Safety
    /// This returns the raw C pointer. Use with caution.
    pub fn as_raw(&self) -> *mut NvOSD_RectParams {
        self.raw
    }

    /// Get the raw pointer as a reference
    ///
    /// # Safety
    /// This returns a reference to the raw C structure. Use with caution.
    pub unsafe fn as_ref(&self) -> &NvOSD_RectParams {
        &*self.raw
    }

    /// Get the raw pointer as a mutable reference
    ///
    /// # Safety
    /// This returns a mutable reference to the raw C structure. Use with caution.
    pub unsafe fn as_mut(&mut self) -> &mut NvOSD_RectParams {
        &mut *self.raw
    }

    /// Convert to the raw C structure
    pub fn to_raw(self) -> NvOSD_RectParams {
        unsafe { *self.raw }
    }

    // Basic properties

    /// Get the left coordinate
    pub fn left(&self) -> f32 {
        unsafe { (*self.raw).left }
    }

    /// Set the left coordinate
    pub fn set_left(&mut self, left: f32) {
        unsafe { (*self.raw).left = left }
    }

    /// Get the top coordinate
    pub fn top(&self) -> f32 {
        unsafe { (*self.raw).top }
    }

    /// Set the top coordinate
    pub fn set_top(&mut self, top: f32) {
        unsafe { (*self.raw).top = top }
    }

    /// Get the width
    pub fn width(&self) -> f32 {
        unsafe { (*self.raw).width }
    }

    /// Set the width
    pub fn set_width(&mut self, width: f32) {
        unsafe { (*self.raw).width = width }
    }

    /// Get the height
    pub fn height(&self) -> f32 {
        unsafe { (*self.raw).height }
    }

    /// Set the height
    pub fn set_height(&mut self, height: f32) {
        unsafe { (*self.raw).height = height }
    }

    // Border properties

    /// Get the border width
    pub fn border_width(&self) -> u32 {
        unsafe { (*self.raw).border_width }
    }

    /// Set the border width
    pub fn set_border_width(&mut self, width: u32) {
        unsafe { (*self.raw).border_width = width as ::std::os::raw::c_uint }
    }

    /// Get whether color info is available
    pub fn has_color_info(&self) -> bool {
        unsafe { (*self.raw).has_color_info != 0 }
    }

    /// Set whether color info is available
    pub fn set_has_color_info(&mut self, has: bool) {
        unsafe { (*self.raw).has_color_info = if has { 1i32 } else { 0i32 } }
    }

    /// Get whether background color is available
    pub fn has_bg_color(&self) -> bool {
        unsafe { (*self.raw).has_bg_color != 0 }
    }

    /// Set whether background color is available
    pub fn set_has_bg_color(&mut self, has: bool) {
        unsafe { (*self.raw).has_bg_color = if has { 1u32 } else { 0u32 } }
    }

    // Color properties

    /// Get the color ID
    pub fn color_id(&self) -> i32 {
        unsafe { (*self.raw).color_id }
    }

    /// Set the color ID
    pub fn set_color_id(&mut self, id: i32) {
        unsafe { (*self.raw).color_id = id }
    }

    /// Get the border color as RGBA
    pub fn border_color(&self) -> [f64; 4] {
        unsafe {
            let color = &(*self.raw).border_color;
            [color.red, color.green, color.blue, color.alpha]
        }
    }

    /// Set the border color as RGBA
    pub fn set_border_color(&mut self, color: [f64; 4]) {
        unsafe {
            (*self.raw).border_color.red = color[0];
            (*self.raw).border_color.green = color[1];
            (*self.raw).border_color.blue = color[2];
            (*self.raw).border_color.alpha = color[3];
        }
    }

    /// Get the background color as RGBA
    pub fn bg_color(&self) -> [f64; 4] {
        unsafe {
            let color = &(*self.raw).bg_color;
            [color.red, color.green, color.blue, color.alpha]
        }
    }

    /// Set the background color as RGBA
    pub fn set_bg_color(&mut self, color: [f64; 4]) {
        unsafe {
            (*self.raw).bg_color.red = color[0];
            (*self.raw).bg_color.green = color[1];
            (*self.raw).bg_color.blue = color[2];
            (*self.raw).bg_color.alpha = color[3];
        }
    }

    // Utility methods

    /// Get the right coordinate
    pub fn right(&self) -> f32 {
        self.left() + self.width()
    }

    /// Get the bottom coordinate
    pub fn bottom(&self) -> f32 {
        self.top() + self.height()
    }

    /// Get the center point
    pub fn center(&self) -> (f32, f32) {
        (
            self.left() + self.width() / 2.0,
            self.top() + self.height() / 2.0,
        )
    }

    /// Get the area
    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }

    /// Check if a point is inside the rectangle
    pub fn contains(&self, x: f32, y: f32) -> bool {
        x >= self.left() && x <= self.right() && y >= self.top() && y <= self.bottom()
    }

    /// Check if this rectangle intersects with another
    pub fn intersects(&self, other: &RectParams) -> bool {
        !(self.right() < other.left()
            || self.left() > other.right()
            || self.bottom() < other.top()
            || self.top() > other.bottom())
    }

    /// Get the intersection with another rectangle
    pub fn intersection(&self, other: &RectParams) -> Option<RectParams> {
        if !self.intersects(other) {
            return None;
        }

        let left = self.left().max(other.left());
        let top = self.top().max(other.top());
        let right = self.right().min(other.right());
        let bottom = self.bottom().min(other.bottom());

        Some(RectParams::new(left, top, right - left, bottom - top))
    }

    /// Get the union with another rectangle
    pub fn union(&self, other: &RectParams) -> RectParams {
        let left = self.left().min(other.left());
        let top = self.top().min(other.top());
        let right = self.right().max(other.right());
        let bottom = self.bottom().max(other.bottom());

        RectParams::new(left, top, right - left, bottom - top)
    }

    /// Scale the rectangle by a factor
    pub fn scale(&mut self, factor: f32) {
        let center = self.center();
        let new_width = self.width() * factor;
        let new_height = self.height() * factor;
        let new_left = center.0 - new_width / 2.0;
        let new_top = center.1 - new_height / 2.0;

        self.set_left(new_left);
        self.set_top(new_top);
        self.set_width(new_width);
        self.set_height(new_height);
    }

    /// Move the rectangle by an offset
    pub fn translate(&mut self, dx: f32, dy: f32) {
        self.set_left(self.left() + dx);
        self.set_top(self.top() + dy);
    }

    /// Expand the rectangle by a margin
    pub fn expand(&mut self, margin: f32) {
        self.set_left(self.left() - margin);
        self.set_top(self.top() - margin);
        self.set_width(self.width() + 2.0 * margin);
        self.set_height(self.height() + 2.0 * margin);
    }

    /// Contract the rectangle by a margin
    pub fn contract(&mut self, margin: f32) {
        self.set_left(self.left() + margin);
        self.set_top(self.top() + margin);
        self.set_width((self.width() - 2.0 * margin).max(0.0));
        self.set_height((self.height() - 2.0 * margin).max(0.0));
    }
}

impl Drop for RectParams {
    fn drop(&mut self) {
        if self.owned && !self.raw.is_null() {
            unsafe {
                let _ = Box::from_raw(self.raw);
            }
        }
    }
}

impl PartialEq for RectParams {
    fn eq(&self, other: &Self) -> bool {
        self.left() == other.left()
            && self.top() == other.top()
            && self.width() == other.width()
            && self.height() == other.height()
    }
}

impl Eq for RectParams {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rect_params_creation() {
        let rect = RectParams::new(10.0, 20.0, 100.0, 200.0);
        assert_eq!(rect.left(), 10.0);
        assert_eq!(rect.top(), 20.0);
        assert_eq!(rect.width(), 100.0);
        assert_eq!(rect.height(), 200.0);
    }

    #[test]
    fn test_rect_params_utility_methods() {
        let rect = RectParams::new(10.0, 20.0, 100.0, 200.0);

        assert_eq!(rect.right(), 110.0);
        assert_eq!(rect.bottom(), 220.0);
        assert_eq!(rect.center(), (60.0, 120.0));
        assert_eq!(rect.area(), 20000.0);
        assert!(rect.contains(60.0, 120.0));
        assert!(!rect.contains(0.0, 0.0));
    }

    #[test]
    fn test_rect_params_operations() {
        let rect1 = RectParams::new(0.0, 0.0, 100.0, 100.0);
        let rect2 = RectParams::new(50.0, 50.0, 100.0, 100.0);

        assert!(rect1.intersects(&rect2));

        let intersection = rect1.intersection(&rect2).unwrap();
        assert_eq!(intersection.left(), 50.0);
        assert_eq!(intersection.top(), 50.0);
        assert_eq!(intersection.width(), 50.0);
        assert_eq!(intersection.height(), 50.0);

        let union = rect1.union(&rect2);
        assert_eq!(union.left(), 0.0);
        assert_eq!(union.top(), 0.0);
        assert_eq!(union.width(), 150.0);
        assert_eq!(union.height(), 150.0);
    }

    #[test]
    fn test_rect_params_transformations() {
        let mut rect = RectParams::new(10.0, 20.0, 100.0, 200.0);

        rect.scale(2.0);
        assert_eq!(rect.width(), 200.0);
        assert_eq!(rect.height(), 400.0);

        rect.translate(10.0, 20.0);
        assert_eq!(rect.left(), -30.0);
        assert_eq!(rect.top(), -60.0);

        rect.expand(5.0);
        assert_eq!(rect.left(), -35.0);
        assert_eq!(rect.top(), -65.0);
        assert_eq!(rect.width(), 210.0);
        assert_eq!(rect.height(), 410.0);
    }
}
