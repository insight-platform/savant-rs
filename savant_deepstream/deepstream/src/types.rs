//! Basic types for DeepStream operations

/// Rectangle parameters for bounding boxes
#[derive(Debug, Clone, Copy)]
pub struct RectParams {
    /// Left coordinate
    pub left: f32,
    /// Top coordinate
    pub top: f32,
    /// Width
    pub width: f32,
    /// Height
    pub height: f32,
}

impl RectParams {
    /// Create a new rectangle with the given dimensions
    pub fn new(left: f32, top: f32, width: f32, height: f32) -> Self {
        Self {
            left,
            top,
            width,
            height,
        }
    }

    /// Create a rectangle from a tuple of (left, top, width, height)
    pub fn from_tuple((left, top, width, height): (f32, f32, f32, f32)) -> Self {
        Self::new(left, top, width, height)
    }

    /// Get the right coordinate
    pub fn right(&self) -> f32 {
        self.left + self.width
    }

    /// Get the bottom coordinate
    pub fn bottom(&self) -> f32 {
        self.top + self.height
    }

    /// Get the center point
    pub fn center(&self) -> (f32, f32) {
        (self.left + self.width / 2.0, self.top + self.height / 2.0)
    }

    /// Check if a point is inside the rectangle
    pub fn contains(&self, x: f32, y: f32) -> bool {
        x >= self.left && x <= self.right() && y >= self.top && y <= self.bottom()
    }

    /// Get the area of the rectangle
    pub fn area(&self) -> f32 {
        self.width * self.height
    }
}

impl From<(f32, f32, f32, f32)> for RectParams {
    fn from((left, top, width, height): (f32, f32, f32, f32)) -> Self {
        Self::new(left, top, width, height)
    }
}

impl From<[f32; 4]> for RectParams {
    fn from([left, top, width, height]: [f32; 4]) -> Self {
        Self::new(left, top, width, height)
    }
}

/// Color parameters for drawing
#[derive(Debug, Clone, Copy)]
pub struct ColorParams {
    /// Red component (0.0 - 1.0)
    pub red: f32,
    /// Green component (0.0 - 1.0)
    pub green: f32,
    /// Blue component (0.0 - 1.0)
    pub blue: f32,
    /// Alpha component (0.0 - 1.0)
    pub alpha: f32,
}

impl ColorParams {
    /// Create a new color
    pub fn new(red: f32, green: f32, blue: f32, alpha: f32) -> Self {
        Self {
            red: red.clamp(0.0, 1.0),
            green: green.clamp(0.0, 1.0),
            blue: blue.clamp(0.0, 1.0),
            alpha: alpha.clamp(0.0, 1.0),
        }
    }

    /// Create a color from RGBA values (0-255)
    pub fn from_rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self::new(
            r as f32 / 255.0,
            g as f32 / 255.0,
            b as f32 / 255.0,
            a as f32 / 255.0,
        )
    }

    /// Create a color from RGB values (0-255)
    pub fn from_rgb(r: u8, g: u8, b: u8) -> Self {
        Self::from_rgba(r, g, b, 255)
    }

    /// Get as RGBA values (0-255)
    pub fn to_rgba(&self) -> (u8, u8, u8, u8) {
        (
            (self.red * 255.0).round() as u8,
            (self.green * 255.0).round() as u8,
            (self.blue * 255.0).round() as u8,
            (self.alpha * 255.0).round() as u8,
        )
    }

    /// Get as RGB values (0-255)
    pub fn to_rgb(&self) -> (u8, u8, u8) {
        let (r, g, b, _) = self.to_rgba();
        (r, g, b)
    }
}

impl Default for ColorParams {
    fn default() -> Self {
        Self::new(1.0, 1.0, 1.0, 1.0) // White
    }
}

impl From<[f32; 4]> for ColorParams {
    fn from([red, green, blue, alpha]: [f32; 4]) -> Self {
        Self::new(red, green, blue, alpha)
    }
}

impl From<[f32; 3]> for ColorParams {
    fn from([red, green, blue]: [f32; 3]) -> Self {
        Self::new(red, green, blue, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rect_params() {
        let rect = RectParams::new(10.0, 20.0, 100.0, 200.0);
        assert_eq!(rect.left, 10.0);
        assert_eq!(rect.top, 20.0);
        assert_eq!(rect.width, 100.0);
        assert_eq!(rect.height, 200.0);
        assert_eq!(rect.right(), 110.0);
        assert_eq!(rect.bottom(), 220.0);
        assert_eq!(rect.center(), (60.0, 120.0));
        assert_eq!(rect.area(), 20000.0);
        assert!(rect.contains(50.0, 100.0));
        assert!(!rect.contains(150.0, 100.0));
    }

    #[test]
    fn test_color_params() {
        let color = ColorParams::new(0.5, 0.25, 0.75, 1.0);
        assert_eq!(color.red, 0.5);
        assert_eq!(color.green, 0.25);
        assert_eq!(color.blue, 0.75);
        assert_eq!(color.alpha, 1.0);

        let color = ColorParams::from_rgb(255, 0, 0);
        assert_eq!(color.red, 1.0);
        assert_eq!(color.green, 0.0);
        assert_eq!(color.blue, 0.0);
        assert_eq!(color.alpha, 1.0);

        let (r, g, b, a) = color.to_rgba();
        assert_eq!(r, 255);
        assert_eq!(g, 0);
        assert_eq!(b, 0);
        assert_eq!(a, 255);
    }
}
