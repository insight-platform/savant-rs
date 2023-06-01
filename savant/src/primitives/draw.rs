use pyo3::prelude::*;

/// Defines the padding for a draw operation.
///
/// # Python API
///
/// The object is read-only in Python API. You may construct it, make a copy
/// or get properties. There is no way to update properties inplace. Fields are
/// not available in Python, use getters.
///
#[pyclass]
#[derive(Clone, Copy, Debug)]
pub struct PaddingDraw {
    pub left: i64,
    pub top: i64,
    pub right: i64,
    pub bottom: i64,
}

#[pymethods]
impl PaddingDraw {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{self:?}")
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Returns a copy of the padding
    ///
    /// # Python API
    ///
    /// ```python
    /// padding.copy()
    /// ```
    ///
    #[pyo3(name = "copy")]
    pub fn copy_py(&self) -> Self {
        *self
    }

    /// Constructs a new padding
    ///
    /// # Python API
    ///
    /// Full initialization:
    ///
    /// ```python
    /// p = PaddingDraw(0, 0, 0, 0)
    /// ```
    /// Default initialization:
    ///
    /// Signature: `PaddingDraw(left=0, top=0, right=0, bottom=0)`
    ///
    /// ```python
    /// p = PaddingDraw()
    /// ```
    ///
    #[new]
    #[pyo3(signature = (left=0, top=0, right=0, bottom=0))]
    pub fn new(left: i64, top: i64, right: i64, bottom: i64) -> Self {
        assert!(left >= 0);
        assert!(top >= 0);
        assert!(right >= 0);
        assert!(bottom >= 0);

        Self {
            left,
            top,
            right,
            bottom,
        }
    }

    #[staticmethod]
    pub fn default_padding() -> Self {
        Self {
            left: 0,
            top: 0,
            right: 0,
            bottom: 0,
        }
    }

    /// Returns the padding as a tuple
    ///
    /// # Python API
    ///
    /// ```python
    /// (l, t, r, b) = padding_spec.padding
    /// ```
    ///
    #[getter]
    pub fn padding(&self) -> (i64, i64, i64, i64) {
        (self.left, self.top, self.right, self.bottom)
    }

    /// Returns the left padding
    ///
    /// # Python API
    ///
    /// ```python
    /// left = padding_spec.left
    /// ```
    ///
    #[getter]
    pub fn left(&self) -> i64 {
        self.left
    }

    /// Returns the top padding
    ///
    /// # Python API
    ///
    /// ```python
    /// top = padding_spec.top
    /// ```
    ///
    #[getter]
    pub fn top(&self) -> i64 {
        self.top
    }

    /// Returns the right padding
    ///
    /// # Python API
    ///
    /// ```python
    /// right = padding_spec.right
    /// ```
    ///
    #[getter]
    pub fn right(&self) -> i64 {
        self.right
    }

    /// Returns the bottom padding
    ///
    /// # Python API
    ///
    /// ```python
    /// bottom = padding_spec.bottom
    /// ```
    ///
    #[getter]
    pub fn bottom(&self) -> i64 {
        self.bottom
    }
}

/// Represents the draw specification for a color.
///
/// # Python API
///
/// The object is read-only in Python API. You may construct it, make a copy
/// or get properties. There is no way to update properties inplace. Fields are
/// not available in Python, use getters.
///
/// Full initialization:
///
/// ```python
/// c = ColorDraw(0, 255, 0, 255)
/// ```
///
/// Default initialization:
///
/// Signature: `ColorDraw(red=0, green=255, blue=0, alpha=255)`
///
/// ```python
/// c = ColorDraw()
/// ```
///
#[pyclass]
#[derive(Clone, Copy, Debug)]
pub struct ColorDraw {
    pub red: i64,
    pub green: i64,
    pub blue: i64,
    pub alpha: i64,
}

#[pymethods]
impl ColorDraw {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{self:?}")
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Returns a copy of the color
    ///
    /// # Python API
    ///
    /// ```python
    /// new_color = color.copy()
    /// ```
    ///
    #[pyo3(name = "copy")]
    pub fn copy_py(&self) -> Self {
        *self
    }

    /// Constructs a new color
    ///
    #[new]
    #[pyo3(signature = (red=0, green=255, blue=0, alpha=255))]
    pub fn new(red: i64, green: i64, blue: i64, alpha: i64) -> Self {
        assert!((0..=255).contains(&red));
        assert!((0..=255).contains(&green));
        assert!((0..=255).contains(&blue));
        assert!((0..=255).contains(&alpha));

        Self {
            red,
            green,
            blue,
            alpha,
        }
    }

    /// Returns the color as a BGRA tuple
    ///
    /// # Python API
    ///
    /// ```python
    /// (b, g, r, a) = color.bgra
    /// ```
    ///
    #[getter]
    pub fn bgra(&self) -> (i64, i64, i64, i64) {
        (self.blue, self.green, self.red, self.alpha)
    }

    /// Returns the color as a RGBA tuple
    ///
    /// # Python API
    ///
    /// ```python
    /// (r, g, b, a) = color.rgba
    /// ```
    ///
    #[getter]
    pub fn rgba(&self) -> (i64, i64, i64, i64) {
        (self.red, self.green, self.blue, self.alpha)
    }

    /// Returns the red component of the color
    ///
    /// # Python API
    ///
    /// ```python
    /// red = color.red
    /// ```
    ///
    #[getter]
    pub fn red(&self) -> i64 {
        self.red
    }

    /// Returns the green component of the color
    ///
    /// # Python API
    ///
    /// ```python
    /// green = color.green
    /// ```
    ///
    #[getter]
    pub fn green(&self) -> i64 {
        self.green
    }

    /// Returns the blue component of the color
    ///
    /// # Python API
    ///
    /// ```python
    /// blue = color.blue
    /// ```
    ///
    #[getter]
    pub fn blue(&self) -> i64 {
        self.blue
    }

    /// Returns the alpha component of the color
    ///
    /// # Python API
    ///
    /// ```python
    /// alpha = color.alpha
    /// ```
    ///
    #[getter]
    pub fn alpha(&self) -> i64 {
        self.alpha
    }

    /// Creates a new transparent color
    ///
    /// # Python API
    ///
    /// ```python
    /// transparent_color = ColorDraw.transparent()
    /// ```
    ///
    #[staticmethod]
    pub fn transparent() -> Self {
        Self::new(0, 0, 0, 0)
    }
}

/// Represents the draw specification for a bounding box.
///
/// # Python API
///
/// The object is read-only in Python API. You may construct it, make a copy
/// or get properties. There is no way to update properties inplace. Fields are
/// not available in Python, use getters.
///
/// Full initialization:
///
/// ```python
/// bb = BoundingBoxDraw(
///     border_color=ColorDraw(0, 255, 0, 255),
///     background_color=ColorDraw(0, 0, 0, 0),
///     thickness=1,
///     padding=PaddingDraw(0, 0, 0, 0))
/// ```
///
/// Default initialization:
///
/// Signature:
/// ```python
/// BoundingBoxDraw(
///     border_color=ColorDraw.transparent(),
///     background_color=ColorDraw.transparent(),
///     thickness=1,
///     padding=PaddingDraw(0, 0, 0, 0))
/// ```
///
/// ```python
/// bb = BoundingBoxDraw()
/// ```
///

#[pyclass]
#[derive(Clone, Copy, Debug)]
pub struct BoundingBoxDraw {
    pub border_color: ColorDraw,
    pub background_color: ColorDraw,
    pub thickness: i64,
    pub padding: PaddingDraw,
}

#[pymethods]
impl BoundingBoxDraw {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{self:?}")
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Returns a copy of the bounding box
    ///
    /// # Python API
    ///
    /// ```python
    /// new_spec = bb_spec.copy()
    /// ```
    ///
    #[pyo3(name = "copy")]
    pub fn copy_py(&self) -> Self {
        *self
    }

    /// Constructs a new bounding box
    ///
    #[new]
    #[pyo3(signature = (
        border_color = ColorDraw::transparent(),
        background_color = ColorDraw::transparent(),
        thickness = 1,
        padding = PaddingDraw::default_padding())
    )]
    pub fn new(
        border_color: ColorDraw,
        background_color: ColorDraw,
        thickness: i64,
        padding: PaddingDraw,
    ) -> Self {
        assert!((0..=100).contains(&thickness));

        Self {
            border_color,
            background_color,
            thickness,
            padding,
        }
    }

    /// Returns the background color of the bounding box
    ///
    /// # Python API
    ///
    /// ```python
    /// border_color = bb_spec.border_color
    /// ```
    ///
    #[getter]
    pub fn border_color(&self) -> ColorDraw {
        self.border_color
    }

    /// Returns the background color of the bounding box
    ///
    /// # Python API
    ///
    /// ```python
    /// background_color = bb_spec.background_color
    /// ```
    ///
    #[getter]
    pub fn background_color(&self) -> ColorDraw {
        self.background_color
    }

    /// Returns the thickness of the bounding box
    ///
    /// # Python API
    ///
    /// ```python
    /// thickness = bb_spec.thickness
    /// ```
    ///
    #[getter]
    pub fn thickness(&self) -> i64 {
        self.thickness
    }

    /// Returns the padding of the bounding box
    ///
    /// # Python API
    ///
    /// ```python
    /// padding = bb_spec.padding
    /// ```
    ///
    #[getter]
    pub fn padding(&self) -> PaddingDraw {
        self.padding
    }
}

/// Represents the draw specification for a central body bullet visualization.
///
/// # Python API
///
/// The object is read-only in Python API. You may construct it, make a copy
/// or get properties. There is no way to update properties inplace. Fields are
/// not available in Python, use getters.
///
/// Full initialization:
///
/// ```python
/// cb = CentralBodyDraw(
///    color=ColorDraw(0, 255, 0, 255),
///    radius=2)
/// ```
///
/// Default initialization:
///
/// Signature:
/// ```python
/// CentralBodyDraw(
///   color,
///   radius=2)
/// ```
///
/// ```python
/// cb = CentralBodyDraw(ColorDraw(255, 0, 0, 255))
/// ```
///
#[pyclass]
#[derive(Clone, Copy, Debug)]
pub struct DotDraw {
    pub color: ColorDraw,
    pub radius: i64,
}

#[pymethods]
impl DotDraw {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{self:?}")
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Returns a copy of the central body
    ///
    /// # Python API
    ///
    /// ```python
    /// new_spec = cb_spec.copy()
    /// ```
    ///
    #[pyo3(name = "copy")]
    pub fn copy_py(&self) -> Self {
        *self
    }

    /// Constructs a new central body bullet specification
    ///
    #[new]
    #[pyo3(signature = (color, radius = 2))]
    pub fn new(color: ColorDraw, radius: i64) -> Self {
        assert!((0..=100).contains(&radius));

        Self { color, radius }
    }

    /// Returns the color of the central body
    ///
    /// # Python API
    ///
    /// ```python
    /// color = cb_spec.color
    /// ```
    ///
    #[getter]
    pub fn color(&self) -> ColorDraw {
        self.color
    }

    /// Returns the radius of the central body
    ///
    /// # Python API
    ///
    /// ```python
    /// radius = cb_spec.radius
    /// ```
    ///
    #[getter]
    pub fn radius(&self) -> i64 {
        self.radius
    }
}

/// Represents the draw specification for a position of a label versus object bounding box.
///
/// # Python API
///
/// The object is read-only in Python API. You may construct it, make a copy
/// or get properties. There is no way to update properties inplace. Fields are
/// not available in Python, use getters.
///
/// Full initialization:
///
/// ```python
/// tli = LabelPositionKind.TopLeftInside
/// tlo = LabelPositionKind.TopLeftOutside
/// c = LabelPositionKind.Center
/// ```
///
#[pyclass]
#[derive(Clone, Copy, Debug)]
pub enum LabelPositionKind {
    /// Margin is relative to the **top** left corner of the text bounding box
    TopLeftInside,
    /// Margin is relative to the **bottom** left corner of the text bounding box
    TopLeftOutside,
    /// Margin is relative to the **top** left corner of the text bounding box
    Center,
}

/// Represents the draw specification for a position of a label versus object bounding box.
///
/// # Python API
///
/// The object is read-only in Python API. You may construct it, make a copy
/// or get properties. There is no way to update properties inplace. Fields are
/// not available in Python, use getters.
///
/// Full initialization:
///
/// ```python
/// lp = LabelPosition(
///   position=LabelPositionKind.TopLeftOutside,
///   margin_x=0,
///   margin_y=-10)
/// ```
///
/// Default initialization:
///
/// Signature:
///
/// ```python
/// LabelPosition(
///  position=LabelPositionKind.TopLeftOutside,
///  margin_x=0,
///  margin_y=-10)
/// ```
///
/// ```python
/// lp = LabelPosition()
/// ```
///
#[pyclass]
#[derive(Clone, Copy, Debug)]
pub struct LabelPosition {
    position: LabelPositionKind,
    margin_x: i64,
    margin_y: i64,
}

#[pymethods]
impl LabelPosition {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{self:?}")
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Returns a copy of the label position
    ///
    /// # Python API
    ///
    /// ```python
    /// new_spec = lp_spec.copy()
    /// ```
    ///
    #[pyo3(name = "copy")]
    pub fn copy_py(&self) -> Self {
        *self
    }

    /// Constructs a new label position specification
    ///
    #[new]
    #[pyo3(signature = (position = LabelPositionKind::TopLeftOutside, margin_x = 0, margin_y = -10))]
    pub fn new(position: LabelPositionKind, margin_x: i64, margin_y: i64) -> Self {
        assert!((-100..=100).contains(&margin_x));
        assert!((-100..=100).contains(&margin_y));

        Self {
            position,
            margin_x,
            margin_y,
        }
    }

    /// Returns the default label position specification
    ///
    /// # Python API
    ///
    /// ```python
    /// lp_spec = LabelPosition.default_position()
    /// ```
    ///
    #[staticmethod]
    pub fn default_position() -> Self {
        Self::new(LabelPositionKind::TopLeftOutside, 0, -10)
    }

    /// Returns the position of the label
    ///
    /// # Python API
    ///
    /// ```python
    /// position = lp_spec.position
    /// ```
    ///
    #[getter]
    pub fn position(&self) -> LabelPositionKind {
        self.position
    }

    /// Returns the margin of the label
    ///
    /// # Python API
    ///
    /// ```python
    /// margin_x = lp_spec.margin_x
    /// ```
    ///
    #[getter]
    pub fn margin_x(&self) -> i64 {
        self.margin_x
    }

    /// Returns the margin of the label
    ///
    /// # Python API
    ///
    /// ```python
    /// margin_y = lp_spec.margin_y
    /// ```
    ///
    #[getter]
    pub fn margin_y(&self) -> i64 {
        self.margin_y
    }
}

/// Represents the draw specification for a label.
///
/// # Python API
///
/// The object is read-only in Python API. You may construct it, make a copy
/// or get properties. There is no way to update properties inplace. Fields are
/// not available in Python, use getters.
///
/// Full initialization:
///
/// ```python
/// ld = LabelDraw(
///   font_color=ColorDraw(255, 255, 255),
///   background_color=ColorDraw(0, 0, 0),
///   border_color=ColorDraw(255, 255, 255),
///   font_scale=0.5,
///   thickness=1,
///   position=LabelPosition(
///     position=LabelPositionKind.TopLeftOutside,
///     margin_x=0,
///     margin_y=-10),
///   padding=PaddingDraw(0, 0, 0, 0),
///   format=["{label}"])
/// ```
///
#[pyclass]
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

#[pymethods]
impl LabelDraw {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{self:?}")
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Returns a copy of the label draw specification
    ///
    /// # Python API
    ///
    /// ```python
    /// new_spec = ld_spec.copy()
    /// ```
    ///
    #[pyo3(name = "copy")]
    pub fn copy_py(&self) -> Self {
        self.clone()
    }

    /// Constructs a new label draw specification
    ///
    #[allow(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (font_color, background_color = ColorDraw::transparent(),
                        border_color = ColorDraw::transparent(), font_scale = 1.0,
                        thickness = 1, position = LabelPosition::default_position(),
                        padding = PaddingDraw::default_padding(),
                        format = vec!["{label}".to_string()])
    )]
    pub fn new(
        font_color: ColorDraw,
        background_color: ColorDraw,
        border_color: ColorDraw,
        font_scale: f64,
        thickness: i64,
        position: LabelPosition,
        padding: PaddingDraw,
        format: Vec<String>,
    ) -> Self {
        assert!((0.0..=200.0).contains(&font_scale));
        assert!((0..=100).contains(&thickness));

        Self {
            font_color,
            background_color,
            border_color,
            font_scale,
            thickness,
            position,
            padding,
            format,
        }
    }

    /// Returns the label font color
    ///
    /// # Python API
    ///
    /// ```python
    /// color = label_draw_spec.font_color
    /// ```
    ///
    #[getter]
    pub fn font_color(&self) -> ColorDraw {
        self.font_color
    }

    /// Returns the background color
    ///
    /// # Python API
    ///
    /// ```python
    /// color = label_draw_spec.background_color
    /// ```
    ///
    #[getter]
    pub fn background_color(&self) -> ColorDraw {
        self.background_color
    }

    /// Returns the border color
    ///
    /// # Python API
    ///
    /// ```python
    /// color = label_draw_spec.border_color
    /// ```
    ///
    #[getter]
    pub fn border_color(&self) -> ColorDraw {
        self.border_color
    }

    /// Returns the font scale
    ///
    /// # Python API
    ///
    /// ```python
    /// font_scale = label_draw_spec.font_scale
    /// ```
    ///
    #[getter]
    pub fn font_scale(&self) -> f64 {
        self.font_scale
    }

    /// Returns the thickness
    ///
    /// # Python API
    ///
    /// ```python
    /// thickness = label_draw_spec.thickness
    /// ```
    ///
    #[getter]
    pub fn thickness(&self) -> i64 {
        self.thickness
    }

    /// Returns the label formatted strings
    ///
    /// # Python API
    ///
    /// ```python
    /// format = label_draw_spec.format
    /// ```
    ///
    #[getter]
    pub fn format(&self) -> Vec<String> {
        self.format.clone()
    }

    #[getter]
    pub fn position(&self) -> LabelPosition {
        self.position
    }

    #[getter]
    pub fn padding(&self) -> PaddingDraw {
        self.padding
    }
}

/// Represents the draw specification for an object.
///
/// # Python API
///
/// The object is read-only in Python API. You may construct it, make a copy
/// or get properties. There is no way to update properties inplace. Fields are
/// not available in Python, use getters.
///
/// Full initialization:
///
/// ```python
/// spec = ObjectDraw(
///     bounding_box=BoundingBoxDraw(
///         border_color=ColorDraw(red=100, blue=50, green=50, alpha=100),
///         background_color=ColorDraw(red=0, blue=50, green=50, alpha=100),
///         thickness=2,
///         padding=PaddingDraw(left=5, top=5, right=5, bottom=5)),
///     label=LabelDraw(
///         font_color=ColorDraw(red=100, blue=50, green=50, alpha=100),
///         border_color=ColorDraw(red=100, blue=50, green=50, alpha=100),
///         background_color=ColorDraw(red=0, blue=50, green=50, alpha=100),
///         padding=PaddingDraw(left=5, top=5, right=5, bottom=5),
///         position=LabelPosition(position=LabelPositionKind.TopLeftOutside, margin_x=0, margin_y=-20),
///         font_scale=2.5,
///         thickness=2,
///         format=["{model}", "{label}", "{confidence}", "{track_id}"]),
///     central_dot=DotDraw(
///         color=ColorDraw(red=100, blue=50, green=50, alpha=100),
///         radius=2),
///     blur=False)
/// ```
///
#[pyclass]
#[derive(Clone, Debug)]
pub struct ObjectDraw {
    pub bounding_box: Option<BoundingBoxDraw>,
    pub central_dot: Option<DotDraw>,
    pub label: Option<LabelDraw>,
    pub blur: bool,
}

#[pymethods]
impl ObjectDraw {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{self:#?}")
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Returns a copy of the object draw specification
    ///
    /// # Python API
    ///
    /// ```python
    /// spec_copy = object_draw_spec.copy()
    /// ```
    ///
    #[pyo3(name = "copy")]
    pub fn copy_py(&self) -> Self {
        self.clone()
    }

    /// Returns the bounding box draw specification
    ///
    #[new]
    #[pyo3(signature = (bounding_box = None, central_dot = None, label = None, blur = false))]
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

    /// Returns blur specification
    ///
    /// # Python API
    ///
    /// ```python
    /// blur = object_draw_spec.blur
    /// ```
    ///
    #[getter]
    pub fn blur(&self) -> bool {
        self.blur
    }

    /// Returns the bounding box draw specification
    ///
    /// # Python API
    ///
    /// ```python
    /// bounding_box = object_draw_spec.bounding_box
    /// ```
    ///
    #[getter]
    pub fn bounding_box(&self) -> Option<BoundingBoxDraw> {
        self.bounding_box
    }

    /// Returns the central dot draw specification
    ///
    /// # Python API
    ///
    /// ```python
    /// central_dot = object_draw_spec.central_dot
    /// ```
    ///
    #[getter]
    pub fn central_dot(&self) -> Option<DotDraw> {
        self.central_dot
    }

    /// Returns the label draw specification
    ///
    /// # Python API
    ///
    /// ```python
    /// label = object_draw_spec.label
    /// ```
    ///
    #[getter]
    pub fn label(&self) -> Option<LabelDraw> {
        self.label.clone()
    }
}

#[derive(Clone, Debug)]
pub enum SetDrawLabelKind {
    OwnLabel(String),
    ParentLabel(String),
}

/// Struct used to specify what label to set for the object. The label can be set
/// for own label or for parent label.
///
/// # Python API
///
/// Own label:
///
/// ```python
/// label = SetDrawLabelKind.own("person")
/// ```
///
/// Parent label:
///
/// ```python
/// label = SetDrawLabelKind.parent("person")
/// ```
///
#[pyclass]
#[derive(Clone, Debug)]
#[pyo3(name = "SetDrawLabelKind")]
pub struct PySetDrawLabelKind {
    pub(crate) inner: SetDrawLabelKind,
}

#[pymethods]
impl PySetDrawLabelKind {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    #[staticmethod]
    pub fn own(label: String) -> Self {
        Self {
            inner: SetDrawLabelKind::OwnLabel(label),
        }
    }

    #[staticmethod]
    pub fn parent(label: String) -> Self {
        Self {
            inner: SetDrawLabelKind::ParentLabel(label),
        }
    }

    pub fn is_own_label(&self) -> bool {
        matches!(self.inner, SetDrawLabelKind::OwnLabel(_))
    }

    pub fn is_parent_label(&self) -> bool {
        matches!(self.inner, SetDrawLabelKind::ParentLabel(_))
    }

    pub fn get_label(&self) -> String {
        match &self.inner {
            SetDrawLabelKind::OwnLabel(label) | SetDrawLabelKind::ParentLabel(label) => {
                label.clone()
            }
        }
    }
}
