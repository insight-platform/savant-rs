use pyo3::prelude::*;

mod callbacks;
pub(crate) mod encoder;
mod engine;
mod error;
mod message;
pub(crate) mod spec;

#[pymodule]
#[pyo3(name = "_native")]
fn picasso_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Encoder types (enums, props, config, properties)
    encoder::register_encoder_classes(m)?;

    // Spec types
    m.add_class::<spec::PyGeneralSpec>()?;
    m.add_class::<spec::PyEvictionDecision>()?;
    m.add_class::<spec::PyConditionalSpec>()?;
    m.add_class::<spec::PyObjectDrawSpec>()?;
    m.add_class::<spec::PyCodecSpec>()?;
    m.add_class::<spec::PySourceSpec>()?;

    // Message types
    m.add_class::<message::PyEncodedOutput>()?;
    m.add_class::<message::PyBypassOutput>()?;

    // Callbacks
    m.add_class::<callbacks::PyCallbacks>()?;

    // Engine
    m.add_class::<engine::PyPicassoEngine>()?;

    Ok(())
}
