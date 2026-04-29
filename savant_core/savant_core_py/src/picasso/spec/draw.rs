use picasso::prelude::ObjectDrawSpec;
use pyo3::prelude::*;
use savant_core::draw as rust_draw;

/// Static per-object draw specifications keyed by `(namespace, label)`.
///
/// Python users insert [`crate::draw_spec::ObjectDraw`] instances via
/// [`insert`]; the inner Rust value is cloned through the `memory_handle`
/// mechanism exposed by `savant_core_py`.
#[pyclass(from_py_object, name = "ObjectDrawSpec", module = "savant_rs.picasso")]
#[derive(Debug, Clone)]
pub struct PyObjectDrawSpec(ObjectDrawSpec);

#[pymethods]
impl PyObjectDrawSpec {
    #[new]
    fn new() -> Self {
        Self(ObjectDrawSpec::new())
    }

    /// Insert a draw specification for the given `(namespace, label)` pair.
    fn insert(&mut self, namespace: &str, label: &str, draw: &crate::draw_spec::ObjectDraw) {
        let ptr = draw.memory_handle() as *const rust_draw::ObjectDraw;
        let rust_draw = unsafe { &*ptr };
        self.0.insert(namespace, label, rust_draw.clone());
    }

    /// Look up the draw spec for an exact `(namespace, label)` match.
    fn lookup(&self, namespace: &str, label: &str) -> Option<crate::draw_spec::ObjectDraw> {
        self.0
            .lookup(namespace, label)
            .map(rebuild_py_object_draw)
    }

    /// Returns `True` if no draw specs have been inserted.
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Number of `(namespace, label)` entries.
    fn len(&self) -> usize {
        self.0.len()
    }

    fn __len__(&self) -> usize {
        self.0.len()
    }

    fn __repr__(&self) -> String {
        format!("ObjectDrawSpec(len={})", self.0.len())
    }
}

impl PyObjectDrawSpec {
    pub(crate) fn to_rust(&self) -> ObjectDrawSpec {
        self.0.clone()
    }

    pub(crate) fn default_empty() -> Self {
        Self(ObjectDrawSpec::new())
    }
}

pub(crate) fn color(c: rust_draw::ColorDraw) -> crate::draw_spec::ColorDraw {
    crate::draw_spec::ColorDraw::new(c.red, c.green, c.blue, c.alpha).unwrap()
}

pub(crate) fn padding(p: rust_draw::PaddingDraw) -> crate::draw_spec::PaddingDraw {
    crate::draw_spec::PaddingDraw::new(p.left, p.top, p.right, p.bottom).unwrap()
}

pub(crate) fn bounding_box(b: rust_draw::BoundingBoxDraw) -> crate::draw_spec::BoundingBoxDraw {
    crate::draw_spec::BoundingBoxDraw::new(
        color(b.border_color),
        color(b.background_color),
        b.thickness,
        padding(b.padding),
    )
    .unwrap()
}

pub(crate) fn dot(d: rust_draw::DotDraw) -> crate::draw_spec::DotDraw {
    crate::draw_spec::DotDraw::new(color(d.color), d.radius).unwrap()
}

pub(crate) fn label_position(p: rust_draw::LabelPosition) -> crate::draw_spec::LabelPosition {
    crate::draw_spec::LabelPosition::new(p.position.into(), p.margin_x, p.margin_y).unwrap()
}

pub(crate) fn label(l: rust_draw::LabelDraw) -> crate::draw_spec::LabelDraw {
    crate::draw_spec::LabelDraw::new(
        color(l.font_color),
        color(l.background_color),
        color(l.border_color),
        l.font_scale,
        l.thickness,
        label_position(l.position),
        padding(l.padding),
        l.format,
    )
    .unwrap()
}

pub(crate) fn rebuild_py_object_draw(d: &rust_draw::ObjectDraw) -> crate::draw_spec::ObjectDraw {
    crate::draw_spec::ObjectDraw::new(
        d.bounding_box.map(bounding_box),
        d.central_dot.map(dot),
        d.label.clone().map(label),
        d.blur,
        d.bbox_source.into(),
    )
}
