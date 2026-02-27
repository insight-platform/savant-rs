//! PyO3 bindings for the `picasso` GPU video processing pipeline.
//!
//! These types are registered in the `savant_rs.picasso` Python submodule
//! by `savant_python` when the `deepstream` feature is enabled.

mod callbacks;
pub(crate) mod encoder;
mod engine;
mod error;
mod message;
pub(crate) mod spec;

use pyo3::prelude::*;

/// Register all Picasso Python classes on the given module.
pub fn register_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    encoder::register_encoder_classes(m)?;

    m.add_class::<spec::PyGeneralSpec>()?;
    m.add_class::<spec::PyEvictionDecision>()?;
    m.add_class::<spec::PyConditionalSpec>()?;
    m.add_class::<spec::PyObjectDrawSpec>()?;
    m.add_class::<spec::PyCodecSpec>()?;
    m.add_class::<spec::PySourceSpec>()?;

    m.add_class::<message::PyEncodedOutput>()?;
    m.add_class::<message::PyBypassOutput>()?;

    m.add_class::<callbacks::PyCallbacks>()?;

    m.add_class::<engine::PyPicassoEngine>()?;

    Ok(())
}
