use anyhow::{bail, Result};

/// Defines the padding for a draw operation.
///
/// The object is read-only after creation in Python. You may construct it, make a copy
/// or get properties. There is no way to update properties inplace. Fields are
/// not available in Python, use getters.
///
///
/// Constructor arguments:
///    left (int): left padding, default 0
///    top (int): top padding, default 0
///    right (int): right padding, default 0
///    bottom (int): bottom padding, default 0
///
/// Returns:
///   The padding object
///
/// .. code-block:: python
///
///   from savant_rs.draw_spec import PaddingDraw
///   padding = PaddingDraw(1, 2, 3, 4)
///
///
#[derive(Clone, Copy, Debug)]
pub struct PaddingDraw {
    pub left: i64,
    pub top: i64,
    pub right: i64,
    pub bottom: i64,
}

impl PaddingDraw {
    pub fn new(left: i64, top: i64, right: i64, bottom: i64) -> Result<Self> {
        if left < 0 || top < 0 || right < 0 || bottom < 0 {
            bail!("Padding values must be greater than or equal to 0",)
        }

        Ok(Self {
            left,
            top,
            right,
            bottom,
        })
    }

    pub fn default_padding() -> Self {
        Self {
            left: 0,
            top: 0,
            right: 0,
            bottom: 0,
        }
    }

    pub fn padding(&self) -> (i64, i64, i64, i64) {
        (self.left, self.top, self.right, self.bottom)
    }

    pub fn left(&self) -> i64 {
        self.left
    }

    pub fn top(&self) -> i64 {
        self.top
    }

    pub fn right(&self) -> i64 {
        self.right
    }

    pub fn bottom(&self) -> i64 {
        self.bottom
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ColorDraw {
    pub red: i64,
    pub green: i64,
    pub blue: i64,
    pub alpha: i64,
}

impl ColorDraw {
    pub fn new(red: i64, green: i64, blue: i64, alpha: i64) -> Result<Self> {
        if !((0..=255).contains(&red)
            && (0..=255).contains(&green)
            && (0..=255).contains(&blue)
            && (0..=255).contains(&alpha))
        {
            bail!("Color values must be greater than or equal to 0",)
        }

        Ok(Self {
            red,
            green,
            blue,
            alpha,
        })
    }

    pub fn bgra(&self) -> (i64, i64, i64, i64) {
        (self.blue, self.green, self.red, self.alpha)
    }

    pub fn rgba(&self) -> (i64, i64, i64, i64) {
        (self.red, self.green, self.blue, self.alpha)
    }

    pub fn red(&self) -> i64 {
        self.red
    }

    pub fn green(&self) -> i64 {
        self.green
    }

    pub fn blue(&self) -> i64 {
        self.blue
    }

    pub fn alpha(&self) -> i64 {
        self.alpha
    }

    pub fn transparent() -> Self {
        Self::new(0, 0, 0, 0).unwrap()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BoundingBoxDraw {
    pub border_color: ColorDraw,
    pub background_color: ColorDraw,
    pub thickness: i64,
    pub padding: PaddingDraw,
}

impl BoundingBoxDraw {
    pub fn new(
        border_color: ColorDraw,
        background_color: ColorDraw,
        thickness: i64,
        padding: PaddingDraw,
    ) -> Result<Self> {
        if !(0..=500).contains(&thickness) {
            bail!("thickness must be >= 0 and <= 500")
        }

        Ok(Self {
            border_color,
            background_color,
            thickness,
            padding,
        })
    }

    pub fn border_color(&self) -> ColorDraw {
        self.border_color
    }
    pub fn background_color(&self) -> ColorDraw {
        self.background_color
    }
    pub fn thickness(&self) -> i64 {
        self.thickness
    }
    pub fn padding(&self) -> PaddingDraw {
        self.padding
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DotDraw {
    pub color: ColorDraw,
    pub radius: i64,
}

impl DotDraw {
    pub fn new(color: ColorDraw, radius: i64) -> Result<Self> {
        if !(0..=100).contains(&radius) {
            bail!("radius must be >= 0 and <= 100")
        }
        Ok(Self { color, radius })
    }
    pub fn color(&self) -> ColorDraw {
        self.color
    }
    pub fn radius(&self) -> i64 {
        self.radius
    }
}
#[derive(Clone, Copy, Debug)]
pub enum LabelPositionKind {
    /// Margin is relative to the **top** left corner of the text bounding box
    TopLeftInside,
    /// Margin is relative to the **bottom** left corner of the text bounding box
    TopLeftOutside,
    /// Margin is relative to the **top** left corner of the text bounding box
    Center,
}
#[derive(Clone, Copy, Debug)]
pub struct LabelPosition {
    position: LabelPositionKind,
    margin_x: i64,
    margin_y: i64,
}
impl LabelPosition {
    pub fn new(position: LabelPositionKind, margin_x: i64, margin_y: i64) -> Result<Self> {
        if !((-100..=100).contains(&margin_x) && (-100..=100).contains(&margin_y)) {
            bail!("margin_x must be >= -100 and <= 100 and margin_y must be >= -100 and <= 100",);
        }

        Ok(Self {
            position,
            margin_x,
            margin_y,
        })
    }
    pub fn default_position() -> Self {
        Self::new(LabelPositionKind::TopLeftOutside, 0, -10).unwrap()
    }
    pub fn position(&self) -> LabelPositionKind {
        self.position
    }
    pub fn margin_x(&self) -> i64 {
        self.margin_x
    }
    pub fn margin_y(&self) -> i64 {
        self.margin_y
    }
}
#[derive(Clone, Debug)]
pub struct LabelDraw {
    pub font_color: ColorDraw,
    pub background_color: ColorDraw,
    pub border_color: ColorDraw,
    pub font_scale: f64,
    pub thickness: i64,
    pub position: LabelPosition,
    pub padding: PaddingDraw,
    pub format: Vec<String>,
}
impl LabelDraw {
    pub fn new(
        font_color: ColorDraw,
        background_color: ColorDraw,
        border_color: ColorDraw,
        font_scale: f64,
        thickness: i64,
        position: LabelPosition,
        padding: PaddingDraw,
        format: Vec<String>,
    ) -> Result<Self> {
        if !((0.0..=200.0).contains(&font_scale) && (0..=100).contains(&thickness)) {
            bail!("font_scale must be >= 0.0 and <= 200.0",);
        }

        Ok(Self {
            font_color,
            background_color,
            border_color,
            font_scale,
            thickness,
            position,
            padding,
            format,
        })
    }

    pub fn font_color(&self) -> ColorDraw {
        self.font_color
    }
    pub fn background_color(&self) -> ColorDraw {
        self.background_color
    }
    pub fn border_color(&self) -> ColorDraw {
        self.border_color
    }
    pub fn font_scale(&self) -> f64 {
        self.font_scale
    }
    pub fn thickness(&self) -> i64 {
        self.thickness
    }
    pub fn format(&self) -> Vec<String> {
        self.format.clone()
    }
    pub fn position(&self) -> LabelPosition {
        self.position
    }
    pub fn padding(&self) -> PaddingDraw {
        self.padding
    }
}
#[derive(Clone, Debug)]
pub struct ObjectDraw {
    pub bounding_box: Option<BoundingBoxDraw>,
    pub central_dot: Option<DotDraw>,
    pub label: Option<LabelDraw>,
    pub blur: bool,
}
impl ObjectDraw {
    pub fn new(
        bounding_box: Option<BoundingBoxDraw>,
        central_dot: Option<DotDraw>,
        label: Option<LabelDraw>,
        blur: bool,
    ) -> Self {
        Self {
            bounding_box,
            central_dot,
            label,
            blur,
        }
    }
    pub fn blur(&self) -> bool {
        self.blur
    }
    pub fn bounding_box(&self) -> Option<BoundingBoxDraw> {
        self.bounding_box
    }
    pub fn central_dot(&self) -> Option<DotDraw> {
        self.central_dot
    }
    pub fn label(&self) -> Option<LabelDraw> {
        self.label.clone()
    }
}

#[derive(Clone, Debug)]
pub enum SetDrawLabelKind {
    OwnLabel(String),
    ParentLabel(String),
}
