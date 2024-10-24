use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use savant_core::draw as rust;

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
pub struct PaddingDraw(pub(crate) rust::PaddingDraw);

#[pymethods]
impl PaddingDraw {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:?}", &self.0)
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
        let pd = rust::PaddingDraw::new(left, top, right, bottom).map_err(|e| {
            PyValueError::new_err(format!(
                "Invalid padding: left={}, top={}, right={}, bottom={}, exception: {}",
                left, top, right, bottom, e
            ))
        })?;

        Ok(Self(pd))
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
        Self(rust::PaddingDraw {
            left: 0,
            top: 0,
            right: 0,
            bottom: 0,
        })
    }

    /// Returns the padding as a tuple
    ///
    /// Returns:
    ///   (left, top, right, bottom)
    ///
    #[getter]
    pub fn padding(&self) -> (i64, i64, i64, i64) {
        (self.0.left, self.0.top, self.0.right, self.0.bottom)
    }

    /// Returns the left padding (int)
    ///
    #[getter]
    pub fn left(&self) -> i64 {
        self.0.left
    }

    /// Returns the top padding (int)
    ///
    #[getter]
    pub fn top(&self) -> i64 {
        self.0.top
    }

    /// Returns the right padding (int)
    ///
    #[getter]
    pub fn right(&self) -> i64 {
        self.0.right
    }

    /// Returns the bottom padding (int)
    ///
    #[getter]
    pub fn bottom(&self) -> i64 {
        self.0.bottom
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
pub struct ColorDraw(rust::ColorDraw);

#[pymethods]
impl ColorDraw {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:?}", &self.0)
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
        let cd = rust::ColorDraw::new(red, green, blue, alpha).map_err(|e| {
            PyValueError::new_err(format!(
                "Invalid color: red={}, green={}, blue={}, alpha={}, exception: {}",
                red, green, blue, alpha, e
            ))
        })?;
        Ok(Self(cd))
    }

    /// The color as a BGRA tuple
    ///
    /// Returns:
    ///   (blue, green, red, alpha)
    ///
    #[getter]
    pub fn bgra(&self) -> (i64, i64, i64, i64) {
        (self.0.blue, self.0.green, self.0.red, self.0.alpha)
    }

    /// The color as a RGBA tuple
    ///
    /// Returns:
    ///   (red, green, blue, alpha)
    ///
    #[getter]
    pub fn rgba(&self) -> (i64, i64, i64, i64) {
        (self.0.red, self.0.green, self.0.blue, self.0.alpha)
    }

    /// The red component of the color (int)
    ///
    #[getter]
    pub fn red(&self) -> i64 {
        self.0.red
    }

    /// The green component of the color (int)
    ///
    #[getter]
    pub fn green(&self) -> i64 {
        self.0.green
    }

    /// The blue component of the color (int)
    ///
    #[getter]
    pub fn blue(&self) -> i64 {
        self.0.blue
    }

    /// The alpha component of the color (int)
    ///
    #[getter]
    pub fn alpha(&self) -> i64 {
        self.0.alpha
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
pub struct BoundingBoxDraw(rust::BoundingBoxDraw);

#[pymethods]
impl BoundingBoxDraw {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:?}", &self.0)
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
        let border_color = border_color.0;
        let background_color = background_color.0;
        let padding = padding.0;
        let bb = rust::BoundingBoxDraw::new(border_color, background_color, thickness, padding)
            .map_err(|e| {
                PyValueError::new_err(format!(
                    "Invalid bounding box: border_color={:?}, background_color={:?}, thickness={}, padding={:?}, exception: {}",
                    border_color, background_color, thickness, padding, e
                ))
            })?;

        Ok(Self(bb))
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
        ColorDraw(self.0.border_color)
    }

    /// Returns the background color of the bounding box
    ///
    #[getter]
    pub fn background_color(&self) -> ColorDraw {
        ColorDraw(self.0.background_color)
    }

    /// Returns the thickness of the bounding box
    ///
    #[getter]
    pub fn thickness(&self) -> i64 {
        self.0.thickness
    }

    /// Returns the padding of the bounding box
    ///
    #[getter]
    pub fn padding(&self) -> PaddingDraw {
        PaddingDraw(self.0.padding)
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
pub struct DotDraw(rust::DotDraw);

#[pymethods]
impl DotDraw {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:?}", &self.0)
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
        let color = color.0;
        let dot_draw = rust::DotDraw::new(color, radius).map_err(|e| {
            PyValueError::new_err(format!(
                "Invalid dot draw: color={:?}, radius={}, exception: {}",
                color, radius, e
            ))
        })?;

        Ok(Self(dot_draw))
    }

    /// Returns the color of the central body
    ///
    #[getter]
    pub fn color(&self) -> ColorDraw {
        ColorDraw(self.0.color)
    }

    /// Returns the radius of the central body
    ///
    #[getter]
    pub fn radius(&self) -> i64 {
        self.0.radius
    }
}

/// Represents the draw specification for a position of a label versus object bounding box.
///
/// The object is read-only after creation in Python. You may construct it, make a copy
/// or get properties. There is no way to update properties inplace. Fields are
/// not available in Python, use getters.
///
#[pyclass(eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LabelPositionKind {
    /// Margin is relative to the **top** left corner of the text bounding box
    TopLeftInside,
    /// Margin is relative to the **bottom** left corner of the text bounding box
    TopLeftOutside,
    /// Margin is relative to the **top** left corner of the text bounding box
    Center,
}

impl From<LabelPositionKind> for rust::LabelPositionKind {
    fn from(lpk: LabelPositionKind) -> Self {
        match lpk {
            LabelPositionKind::TopLeftInside => rust::LabelPositionKind::TopLeftInside,
            LabelPositionKind::TopLeftOutside => rust::LabelPositionKind::TopLeftOutside,
            LabelPositionKind::Center => rust::LabelPositionKind::Center,
        }
    }
}

impl From<rust::LabelPositionKind> for LabelPositionKind {
    fn from(lpk: rust::LabelPositionKind) -> Self {
        match lpk {
            rust::LabelPositionKind::TopLeftInside => LabelPositionKind::TopLeftInside,
            rust::LabelPositionKind::TopLeftOutside => LabelPositionKind::TopLeftOutside,
            rust::LabelPositionKind::Center => LabelPositionKind::Center,
        }
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
pub struct LabelPosition(rust::LabelPosition);

#[pymethods]
impl LabelPosition {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:?}", &self.0)
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
        let position = rust::LabelPosition::new(position.into(), margin_x, margin_y)
            .map_err(|e| PyValueError::new_err(format!("Invalid label position: {:?}", e)))?;

        Ok(Self(position))
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
        self.0.position.into()
    }

    /// Returns the margin of the label
    ///
    #[getter]
    pub fn margin_x(&self) -> i64 {
        self.0.margin_x
    }

    /// Returns the margin of the label
    ///
    #[getter]
    pub fn margin_y(&self) -> i64 {
        self.0.margin_y
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
pub struct LabelDraw(rust::LabelDraw);

#[pymethods]
impl LabelDraw {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:?}", &self.0)
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
        let font_color = font_color.0;
        let background_color = background_color.0;
        let border_color = border_color.0;
        let position = position.0;
        let padding = padding.0;
        let label_draw = rust::LabelDraw::new(
            font_color,
            background_color,
            border_color,
            font_scale,
            thickness,
            position,
            padding,
            format,
        )
        .map_err(|e| PyValueError::new_err(format!("Invalid label draw: {:?}", e)))?;

        Ok(Self(label_draw))
    }

    /// Returns the label font color
    ///
    #[getter]
    pub fn font_color(&self) -> ColorDraw {
        ColorDraw(self.0.font_color)
    }

    /// Returns the background color
    ///
    #[getter]
    pub fn background_color(&self) -> ColorDraw {
        ColorDraw(self.0.background_color)
    }

    /// Returns the border color
    ///
    #[getter]
    pub fn border_color(&self) -> ColorDraw {
        ColorDraw(self.0.border_color)
    }

    /// Returns the font cloud
    ///
    #[getter]
    pub fn font_scale(&self) -> f64 {
        self.0.font_scale
    }

    /// Returns the thickness
    ///
    #[getter]
    pub fn thickness(&self) -> i64 {
        self.0.thickness
    }

    /// Returns the label formatted strings
    ///
    #[getter]
    pub fn format(&self) -> Vec<String> {
        self.0.format.clone()
    }

    #[getter]
    pub fn position(&self) -> LabelPosition {
        LabelPosition(self.0.position)
    }

    #[getter]
    pub fn padding(&self) -> PaddingDraw {
        PaddingDraw(self.0.padding)
    }
}

/// Represents the draw specification for an object.
///
#[pyclass]
#[derive(Clone, Debug)]
pub struct ObjectDraw(rust::ObjectDraw);

#[pymethods]
impl ObjectDraw {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:?}", &self.0)
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
        let bounding_box = bounding_box.map(|x| x.0);
        let central_dot = central_dot.map(|x| x.0);
        let label = label.map(|x| x.0);
        let object_draw = rust::ObjectDraw::new(bounding_box, central_dot, label, blur);
        Self(object_draw)
    }

    /// Returns blur specification
    ///
    #[getter]
    pub fn blur(&self) -> bool {
        self.0.blur
    }

    /// Returns the bounding box draw specification
    ///
    #[getter]
    pub fn bounding_box(&self) -> Option<BoundingBoxDraw> {
        self.0.bounding_box.map(BoundingBoxDraw)
    }

    /// Returns the central dot draw specification
    ///
    #[getter]
    pub fn central_dot(&self) -> Option<DotDraw> {
        self.0.central_dot.map(DotDraw)
    }

    /// Returns the label draw specification
    ///
    #[getter]
    pub fn label(&self) -> Option<LabelDraw> {
        self.0.label.clone().map(LabelDraw)
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct SetDrawLabelKind(pub(crate) rust::DrawLabelKind);

#[pymethods]
impl SetDrawLabelKind {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:?}", &self.0)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    #[staticmethod]
    pub fn own(label: String) -> Self {
        Self(rust::DrawLabelKind::OwnLabel(label))
    }

    #[staticmethod]
    pub fn parent(label: String) -> Self {
        Self(rust::DrawLabelKind::ParentLabel(label))
    }

    pub fn is_own_label(&self) -> bool {
        matches!(self.0, rust::DrawLabelKind::OwnLabel(_))
    }

    pub fn is_parent_label(&self) -> bool {
        matches!(self.0, rust::DrawLabelKind::ParentLabel(_))
    }

    pub fn get_label(&self) -> String {
        match &self.0 {
            rust::DrawLabelKind::OwnLabel(label) | rust::DrawLabelKind::ParentLabel(label) => {
                label.clone()
            }
        }
    }
}
