use picasso::prelude::SourceSpec;
use pyo3::prelude::*;

use super::codec::PyCodecSpec;
use super::conditional::PyConditionalSpec;
use super::draw::PyObjectDrawSpec;

/// Complete per-source configuration combining all spec facets.
#[pyclass(from_py_object, name = "SourceSpec", module = "savant_rs.picasso")]
#[derive(Debug, Clone)]
pub struct PySourceSpec {
    codec: PyCodecSpec,
    conditional: PyConditionalSpec,
    draw: PyObjectDrawSpec,
    #[pyo3(get, set)]
    pub font_family: String,
    #[pyo3(get, set)]
    pub idle_timeout_secs: Option<u64>,
    #[pyo3(get, set)]
    pub use_on_render: bool,
    #[pyo3(get, set)]
    pub use_on_gpumat: bool,
}

#[pymethods]
impl PySourceSpec {
    #[new]
    #[pyo3(signature = (
        codec = None,
        conditional = None,
        draw = None,
        font_family = "sans-serif".to_string(),
        idle_timeout_secs = None,
        use_on_render = false,
        use_on_gpumat = false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        codec: Option<PyCodecSpec>,
        conditional: Option<PyConditionalSpec>,
        draw: Option<PyObjectDrawSpec>,
        font_family: String,
        idle_timeout_secs: Option<u64>,
        use_on_render: bool,
        use_on_gpumat: bool,
    ) -> Self {
        Self {
            codec: codec.unwrap_or_else(PyCodecSpec::default_drop),
            conditional: conditional.unwrap_or_default(),
            draw: draw.unwrap_or_else(PyObjectDrawSpec::default_empty),
            font_family,
            idle_timeout_secs,
            use_on_render,
            use_on_gpumat,
        }
    }

    /// What to do with frames (drop / bypass / encode).
    #[getter]
    fn get_codec(&self) -> PyCodecSpec {
        self.codec.clone()
    }

    #[setter]
    fn set_codec(&mut self, codec: PyCodecSpec) {
        self.codec = codec;
    }

    /// Attribute gates for conditional processing.
    #[getter]
    fn get_conditional(&self) -> PyConditionalSpec {
        self.conditional.clone()
    }

    #[setter]
    fn set_conditional(&mut self, conditional: PyConditionalSpec) {
        self.conditional = conditional;
    }

    /// Static draw specs for object overlays.
    #[getter]
    fn get_draw(&self) -> PyObjectDrawSpec {
        self.draw.clone()
    }

    #[setter]
    fn set_draw(&mut self, draw: PyObjectDrawSpec) {
        self.draw = draw;
    }

    fn __repr__(&self) -> String {
        format!(
            "SourceSpec(codec={:?}, conditional={:?}, draw={:?}, \
             font_family={:?}, idle_timeout_secs={:?}, \
             use_on_render={}, use_on_gpumat={})",
            self.codec,
            self.conditional,
            self.draw,
            self.font_family,
            self.idle_timeout_secs,
            self.use_on_render,
            self.use_on_gpumat,
        )
    }
}

impl PySourceSpec {
    pub(crate) fn to_rust(&self) -> SourceSpec {
        SourceSpec {
            codec: self.codec.to_rust(),
            conditional: self.conditional.to_rust(),
            draw: self.draw.to_rust(),
            font_family: self.font_family.clone(),
            idle_timeout_secs: self.idle_timeout_secs,
            use_on_render: self.use_on_render,
            use_on_gpumat: self.use_on_gpumat,
        }
    }
}
