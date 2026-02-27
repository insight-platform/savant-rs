use picasso::prelude::ConditionalSpec;
use pyo3::prelude::*;

/// Attribute-based gates for conditional processing.
///
/// When set, the pipeline checks whether the frame carries the specified
/// attribute before proceeding with the corresponding stage.
#[pyclass(from_py_object, name = "ConditionalSpec", module = "savant_rs.picasso")]
#[derive(Debug, Clone, Default)]
pub struct PyConditionalSpec {
    encode_attribute: Option<(String, String)>,
    render_attribute: Option<(String, String)>,
}

#[pymethods]
impl PyConditionalSpec {
    #[new]
    #[pyo3(signature = (encode_attribute = None, render_attribute = None))]
    fn new(
        encode_attribute: Option<(String, String)>,
        render_attribute: Option<(String, String)>,
    ) -> Self {
        Self {
            encode_attribute,
            render_attribute,
        }
    }

    /// Attribute `(namespace, name)` the frame must carry to be encoded.
    /// `None` means unconditional.
    #[getter]
    fn get_encode_attribute(&self) -> Option<(String, String)> {
        self.encode_attribute.clone()
    }

    #[setter]
    fn set_encode_attribute(&mut self, value: Option<(String, String)>) {
        self.encode_attribute = value;
    }

    /// Attribute `(namespace, name)` the frame must carry for the Skia
    /// rendering stage to run.  `None` means unconditional.
    #[getter]
    fn get_render_attribute(&self) -> Option<(String, String)> {
        self.render_attribute.clone()
    }

    #[setter]
    fn set_render_attribute(&mut self, value: Option<(String, String)>) {
        self.render_attribute = value;
    }

    fn __repr__(&self) -> String {
        format!(
            "ConditionalSpec(encode_attribute={:?}, render_attribute={:?})",
            self.encode_attribute, self.render_attribute,
        )
    }
}

impl PyConditionalSpec {
    pub(crate) fn to_rust(&self) -> ConditionalSpec {
        ConditionalSpec {
            encode_attribute: self.encode_attribute.clone(),
            render_attribute: self.render_attribute.clone(),
        }
    }
}
