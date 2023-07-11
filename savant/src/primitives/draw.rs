use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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

    /// Returns a copy of the padding object
    ///
    /// Returns:
    ///   The padding object
    ///
    /// .. code-block:: python
    ///
    ///   from savant_rs.draw_spec import PaddingDraw
    ///   padding = PaddingDraw(1, 2, 3, 4)
    ///   padding_copy = padding.copy()
    ///
    #[pyo3(name = "copy")]
    pub fn copy_py(&self) -> Self {
        *self
    }

    #[new]
    #[pyo3(signature = (left=0, top=0, right=0, bottom=0))]
    pub fn new(left: i64, top: i64, right: i64, bottom: i64) -> PyResult<Self> {
        if left < 0 || top < 0 || right < 0 || bottom < 0 {
            return Err(PyValueError::new_err(
                "Padding values must be greater than or equal to 0",
            ));
        }

        Ok(Self {
            left,
            top,
            right,
            bottom,
        })
    }

    /// Creates a new padding object with all fields set to 0
    ///
    /// Returns:
    ///   The padding object
    ///
    /// .. code-block:: python
    ///
    ///   from savant_rs.draw_spec import PaddingDraw
    ///   padding = PaddingDraw.default_padding()
    ///
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
    /// Returns:
    ///   (left, top, right, bottom)
    ///
    #[getter]
    pub fn padding(&self) -> (i64, i64, i64, i64) {
        (self.left, self.top, self.right, self.bottom)
    }

    /// Returns the left padding (int)
    ///
    #[getter]
    pub fn left(&self) -> i64 {
        self.left
    }

    /// Returns the top padding (int)
    ///
    #[getter]
    pub fn top(&self) -> i64 {
        self.top
    }

    /// Returns the right padding (int)
    ///
    #[getter]
    pub fn right(&self) -> i64 {
        self.right
    }

    /// Returns the bottom padding (int)
    ///
    #[getter]
    pub fn bottom(&self) -> i64 {
        self.bottom
    }
}

/// Represents the draw specification for a color.
///
/// The object is read-only after creation in Python. You may construct it, make a copy
/// or get properties. There is no way to update properties inplace. Fields are
/// not available in Python, use getters.
///
/// Constructor arguments:
///   red (int): red component, default 0
///   green (int): green component, default 255
///   blue (int): blue component, default 0
///   alpha (int): alpha component, default 255
///
/// Returns:
///   The color object
///
/// .. code-block:: python
///
///   from savant_rs.draw_spec import ColorDraw
///   color = ColorDraw(1, 2, 3, 4)
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
    #[pyo3(name = "copy")]
    pub fn copy_py(&self) -> Self {
        *self
    }

    /// Constructs a new color
    ///
    #[new]
    #[pyo3(signature = (red=0, green=255, blue=0, alpha=255))]
    pub fn new(red: i64, green: i64, blue: i64, alpha: i64) -> PyResult<Self> {
        if !((0..=255).contains(&red)
            && (0..=255).contains(&green)
            && (0..=255).contains(&blue)
            && (0..=255).contains(&alpha))
        {
            return Err(PyValueError::new_err(
                "Color values must be greater than or equal to 0",
            ));
        }

        Ok(Self {
            red,
            green,
            blue,
            alpha,
        })
    }

    /// The color as a BGRA tuple
    ///
    /// Returns:
    ///   (blue, green, red, alpha)
    ///
    #[getter]
    pub fn bgra(&self) -> (i64, i64, i64, i64) {
        (self.blue, self.green, self.red, self.alpha)
    }

    /// The color as a RGBA tuple
    ///
    /// Returns:
    ///   (red, green, blue, alpha)
    ///
    #[getter]
    pub fn rgba(&self) -> (i64, i64, i64, i64) {
        (self.red, self.green, self.blue, self.alpha)
    }

    /// The red component of the color (int)
    ///
    #[getter]
    pub fn red(&self) -> i64 {
        self.red
    }

    /// The green component of the color (int)
    ///
    #[getter]
    pub fn green(&self) -> i64 {
        self.green
    }

    /// The blue component of the color (int)
    ///
    #[getter]
    pub fn blue(&self) -> i64 {
        self.blue
    }

    /// The alpha component of the color (int)
    ///
    #[getter]
    pub fn alpha(&self) -> i64 {
        self.alpha
    }

    /// Creates a new transparent color
    ///
    /// Returns:
    ///   The color object
    ///
    /// .. code-block:: python
    ///
    ///   from savant_rs.draw_spec import ColorDraw
    ///   color = ColorDraw.transparent()
    ///
    #[staticmethod]
    pub fn transparent() -> Self {
        Self::new(0, 0, 0, 0).unwrap()
    }
}

/// Represents the draw specification for a bounding box.
///
/// The object is read-only after creation in Python. You may construct it, make a copy
/// or get properties. There is no way to update properties inplace. Fields are
/// not available in Python, use getters.
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
        thickness = 2,
        padding = PaddingDraw::default_padding())
    )]
    pub fn new(
        border_color: ColorDraw,
        background_color: ColorDraw,
        thickness: i64,
        padding: PaddingDraw,
    ) -> PyResult<Self> {
        if !(0..=500).contains(&thickness) {
            return Err(PyValueError::new_err("thickness must be >= 0 and <= 500"));
        }

        Ok(Self {
            border_color,
            background_color,
            thickness,
            padding,
        })
    }

    /// Returns the border color of the bounding box
    ///
    /// Returns:
    ///   The `ColorDraw` object
    ///
    /// .. code-block:: python
    ///
    ///   from savant_rs.draw_spec import ColorDraw, PaddingDraw, BoundingBoxDraw
    ///   border_color = ColorDraw(255, 255, 255, 255)
    ///   background_color = ColorDraw(0, 0, 0, 0)
    ///   box = BoundingBoxDraw(border_color, background_color, 1, PaddingDraw(1, 2, 3, 4))
    ///   border_color = box.border_color
    ///
    #[getter]
    pub fn border_color(&self) -> ColorDraw {
        self.border_color
    }

    /// Returns the background color of the bounding box
    ///
    #[getter]
    pub fn background_color(&self) -> ColorDraw {
        self.background_color
    }

    /// Returns the thickness of the bounding box
    ///
    #[getter]
    pub fn thickness(&self) -> i64 {
        self.thickness
    }

    /// Returns the padding of the bounding box
    ///
    #[getter]
    pub fn padding(&self) -> PaddingDraw {
        self.padding
    }
}

/// Represents the draw specification for a central body bullet visualization.
///
/// The object is read-only after creation in Python. You may construct it, make a copy
/// or get properties. There is no way to update properties inplace. Fields are
/// not available in Python, use getters.
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
    #[pyo3(name = "copy")]
    pub fn copy_py(&self) -> Self {
        *self
    }

    /// Constructs a new central body bullet specification
    ///
    #[new]
    #[pyo3(signature = (color, radius = 2))]
    pub fn new(color: ColorDraw, radius: i64) -> PyResult<Self> {
        if !(0..=100).contains(&radius) {
            return Err(PyValueError::new_err("radius must be >= 0 and <= 100"));
        }
        Ok(Self { color, radius })
    }

    /// Returns the color of the central body
    ///
    #[getter]
    pub fn color(&self) -> ColorDraw {
        self.color
    }

    /// Returns the radius of the central body
    ///
    #[getter]
    pub fn radius(&self) -> i64 {
        self.radius
    }
}

/// Represents the draw specification for a position of a label versus object bounding box.
///
/// The object is read-only after creation in Python. You may construct it, make a copy
/// or get properties. There is no way to update properties inplace. Fields are
/// not available in Python, use getters.
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
/// The object is read-only after creation in Python. You may construct it, make a copy
/// or get properties. There is no way to update properties inplace. Fields are
/// not available in Python, use getters.
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
    #[pyo3(name = "copy")]
    pub fn copy_py(&self) -> Self {
        *self
    }

    /// Constructs a new label position specification
    ///
    #[new]
    #[pyo3(signature = (position = LabelPositionKind::TopLeftOutside, margin_x = 0, margin_y = -10))]
    pub fn new(position: LabelPositionKind, margin_x: i64, margin_y: i64) -> PyResult<Self> {
        if !((-100..=100).contains(&margin_x) && (-100..=100).contains(&margin_y)) {
            return Err(PyValueError::new_err(
                "margin_x must be >= -100 and <= 100 and margin_y must be >= -100 and <= 100",
            ));
        }

        Ok(Self {
            position,
            margin_x,
            margin_y,
        })
    }

    /// Returns the default label position specification
    ///
    #[staticmethod]
    pub fn default_position() -> Self {
        Self::new(LabelPositionKind::TopLeftOutside, 0, -10).unwrap()
    }

    /// Returns the position of the label
    ///
    #[getter]
    pub fn position(&self) -> LabelPositionKind {
        self.position
    }

    /// Returns the margin of the label
    ///
    #[getter]
    pub fn margin_x(&self) -> i64 {
        self.margin_x
    }

    /// Returns the margin of the label
    ///
    #[getter]
    pub fn margin_y(&self) -> i64 {
        self.margin_y
    }
}

/// Represents the draw specification for a label.
///
/// The object is read-only after creation in Python. You may construct it, make a copy
/// or get properties. There is no way to update properties inplace. Fields are
/// not available in Python, use getters.
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
    ) -> PyResult<Self> {
        if !((0.0..=200.0).contains(&font_scale) && (0..=100).contains(&thickness)) {
            return Err(PyValueError::new_err(
                "font_scale must be >= 0.0 and <= 200.0",
            ));
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

    /// Returns the label font color
    ///
    #[getter]
    pub fn font_color(&self) -> ColorDraw {
        self.font_color
    }

    /// Returns the background color
    ///
    #[getter]
    pub fn background_color(&self) -> ColorDraw {
        self.background_color
    }

    /// Returns the border color
    ///
    #[getter]
    pub fn border_color(&self) -> ColorDraw {
        self.border_color
    }

    /// Returns the font scale
    ///
    #[getter]
    pub fn font_scale(&self) -> f64 {
        self.font_scale
    }

    /// Returns the thickness
    ///
    #[getter]
    pub fn thickness(&self) -> i64 {
        self.thickness
    }

    /// Returns the label formatted strings
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
    #[getter]
    pub fn blur(&self) -> bool {
        self.blur
    }

    /// Returns the bounding box draw specification
    ///
    #[getter]
    pub fn bounding_box(&self) -> Option<BoundingBoxDraw> {
        self.bounding_box
    }

    /// Returns the central dot draw specification
    ///
    #[getter]
    pub fn central_dot(&self) -> Option<DotDraw> {
        self.central_dot
    }

    /// Returns the label draw specification
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
#[pyclass]
#[derive(Clone, Debug)]
#[pyo3(name = "SetDrawLabelKind")]
pub struct SetDrawLabelKindProxy {
    pub(crate) inner: SetDrawLabelKind,
}

#[pymethods]
impl SetDrawLabelKindProxy {
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
