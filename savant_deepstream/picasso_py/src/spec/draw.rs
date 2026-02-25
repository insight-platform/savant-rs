use picasso::prelude::ObjectDrawSpec;
use pyo3::prelude::*;
use savant_core::draw as rust_draw;

/// Static per-object draw specifications keyed by `(namespace, label)`.
///
/// Python users insert [`savant_core_py::draw_spec::ObjectDraw`] instances via
/// [`insert`]; the inner Rust value is cloned through the `memory_handle`
/// mechanism exposed by `savant_core_py`.
#[pyclass(from_py_object, name = "ObjectDrawSpec", module = "picasso._native")]
#[derive(Debug, Clone)]
pub struct PyObjectDrawSpec {
    inner: ObjectDrawSpec,
}

#[pymethods]
impl PyObjectDrawSpec {
    #[new]
    fn new() -> Self {
        Self {
            inner: ObjectDrawSpec::new(),
        }
    }

    /// Insert a draw specification for the given `(namespace, label)` pair.
    ///
    /// `draw` is a `savant_core.draw_spec.ObjectDraw` instance.  Its inner
    /// Rust value is cloned into this map via the `memory_handle` pointer.
    fn insert(
        &mut self,
        namespace: &str,
        label: &str,
        draw: &savant_core_py::draw_spec::ObjectDraw,
    ) {
        // SAFETY: `memory_handle()` returns a pointer to the live inner
        // `savant_core::draw::ObjectDraw` owned by the Python wrapper.  The
        // reference is only used for the duration of this call while `draw`
        // is borrowed by PyO3, so the pointer is guaranteed to be valid.
        let ptr = draw.memory_handle() as *const rust_draw::ObjectDraw;
        let rust_draw = unsafe { &*ptr };
        self.inner.insert(namespace, label, rust_draw.clone());
    }

    /// Look up the draw spec for an exact `(namespace, label)` match.
    ///
    /// Returns `None` if no entry exists.
    fn lookup(
        &self,
        namespace: &str,
        label: &str,
    ) -> Option<savant_core_py::draw_spec::ObjectDraw> {
        self.inner
            .lookup(namespace, label)
            .map(rebuild_py_object_draw)
    }

    /// Returns `True` if no draw specs have been inserted.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Number of `(namespace, label)` entries.
    fn len(&self) -> usize {
        self.inner.len()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("ObjectDrawSpec(len={})", self.inner.len())
    }
}

impl PyObjectDrawSpec {
    pub(crate) fn to_rust(&self) -> ObjectDrawSpec {
        self.inner.clone()
    }

    /// Convenience for constructing an empty spec from Rust.
    pub(crate) fn default_empty() -> Self {
        Self {
            inner: ObjectDrawSpec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Reconstruction helpers
//
// `savant_core_py` wrapper newtypes keep their inner fields private, so we
// must round-trip through the public Python-facing constructors.  All data
// originates from validated Rust structs, so the `.unwrap()` calls below are
// safe.
// ---------------------------------------------------------------------------

pub(crate) fn color(c: rust_draw::ColorDraw) -> savant_core_py::draw_spec::ColorDraw {
    savant_core_py::draw_spec::ColorDraw::new(c.red, c.green, c.blue, c.alpha).unwrap()
}

pub(crate) fn padding(p: rust_draw::PaddingDraw) -> savant_core_py::draw_spec::PaddingDraw {
    savant_core_py::draw_spec::PaddingDraw::new(p.left, p.top, p.right, p.bottom).unwrap()
}

pub(crate) fn bounding_box(
    b: rust_draw::BoundingBoxDraw,
) -> savant_core_py::draw_spec::BoundingBoxDraw {
    savant_core_py::draw_spec::BoundingBoxDraw::new(
        color(b.border_color),
        color(b.background_color),
        b.thickness,
        padding(b.padding),
    )
    .unwrap()
}

pub(crate) fn dot(d: rust_draw::DotDraw) -> savant_core_py::draw_spec::DotDraw {
    savant_core_py::draw_spec::DotDraw::new(color(d.color), d.radius).unwrap()
}

pub(crate) fn label_position(
    p: rust_draw::LabelPosition,
) -> savant_core_py::draw_spec::LabelPosition {
    savant_core_py::draw_spec::LabelPosition::new(p.position.into(), p.margin_x, p.margin_y)
        .unwrap()
}

pub(crate) fn label(l: rust_draw::LabelDraw) -> savant_core_py::draw_spec::LabelDraw {
    savant_core_py::draw_spec::LabelDraw::new(
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

pub(crate) fn rebuild_py_object_draw(
    d: &rust_draw::ObjectDraw,
) -> savant_core_py::draw_spec::ObjectDraw {
    savant_core_py::draw_spec::ObjectDraw::new(
        d.bounding_box.map(bounding_box),
        d.central_dot.map(dot),
        d.label.clone().map(label),
        d.blur,
        d.bbox_source.into(),
    )
}
