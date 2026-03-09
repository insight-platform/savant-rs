use picasso::prelude::{CallbackInvocationOrder, SourceSpec};
use pyo3::prelude::*;

use super::codec::PyCodecSpec;
use super::conditional::PyConditionalSpec;
use super::draw::PyObjectDrawSpec;

/// Controls when the `on_gpumat` callback fires relative to Skia rendering.
#[pyclass(
    from_py_object,
    name = "CallbackInvocationOrder",
    module = "savant_rs.picasso",
    eq,
    hash,
    frozen
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PyCallbackInvocationOrder {
    /// Skia render → `on_gpumat` (default).
    #[pyo3(name = "SkiaGpuMat")]
    #[default]
    SkiaGpuMat,
    /// `on_gpumat` → Skia render.
    #[pyo3(name = "GpuMatSkia")]
    GpuMatSkia,
    /// `on_gpumat` → Skia render → `on_gpumat`.
    #[pyo3(name = "GpuMatSkiaGpuMat")]
    GpuMatSkiaGpuMat,
}

#[pymethods]
impl PyCallbackInvocationOrder {
    /// Create from string name.
    #[staticmethod]
    fn from_name(name: &str) -> PyResult<Self> {
        match name {
            "SkiaGpuMat" => Ok(Self::SkiaGpuMat),
            "GpuMatSkia" => Ok(Self::GpuMatSkia),
            "GpuMatSkiaGpuMat" => Ok(Self::GpuMatSkiaGpuMat),
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown CallbackInvocationOrder: {name}"
            ))),
        }
    }

    fn __repr__(&self) -> String {
        match self {
            Self::SkiaGpuMat => "CallbackInvocationOrder.SkiaGpuMat".to_string(),
            Self::GpuMatSkia => "CallbackInvocationOrder.GpuMatSkia".to_string(),
            Self::GpuMatSkiaGpuMat => "CallbackInvocationOrder.GpuMatSkiaGpuMat".to_string(),
        }
    }
}

impl From<PyCallbackInvocationOrder> for CallbackInvocationOrder {
    fn from(val: PyCallbackInvocationOrder) -> Self {
        match val {
            PyCallbackInvocationOrder::SkiaGpuMat => CallbackInvocationOrder::SkiaGpuMat,
            PyCallbackInvocationOrder::GpuMatSkia => CallbackInvocationOrder::GpuMatSkia,
            PyCallbackInvocationOrder::GpuMatSkiaGpuMat => {
                CallbackInvocationOrder::GpuMatSkiaGpuMat
            }
        }
    }
}

impl From<CallbackInvocationOrder> for PyCallbackInvocationOrder {
    fn from(val: CallbackInvocationOrder) -> Self {
        match val {
            CallbackInvocationOrder::SkiaGpuMat => PyCallbackInvocationOrder::SkiaGpuMat,
            CallbackInvocationOrder::GpuMatSkia => PyCallbackInvocationOrder::GpuMatSkia,
            CallbackInvocationOrder::GpuMatSkiaGpuMat => {
                PyCallbackInvocationOrder::GpuMatSkiaGpuMat
            }
        }
    }
}

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
    #[pyo3(get, set)]
    pub callback_order: PyCallbackInvocationOrder,
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
        callback_order = PyCallbackInvocationOrder::SkiaGpuMat,
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
        callback_order: PyCallbackInvocationOrder,
    ) -> Self {
        Self {
            codec: codec.unwrap_or_else(PyCodecSpec::default_drop),
            conditional: conditional.unwrap_or_default(),
            draw: draw.unwrap_or_else(PyObjectDrawSpec::default_empty),
            font_family,
            idle_timeout_secs,
            use_on_render,
            use_on_gpumat,
            callback_order,
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
             use_on_render={}, use_on_gpumat={}, callback_order={:?})",
            self.codec,
            self.conditional,
            self.draw,
            self.font_family,
            self.idle_timeout_secs,
            self.use_on_render,
            self.use_on_gpumat,
            self.callback_order,
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
            callback_order: self.callback_order.into(),
        }
    }
}
