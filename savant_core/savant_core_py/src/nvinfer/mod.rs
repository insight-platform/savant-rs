//! PyO3 bindings for the `nvinfer` crate.
//!
//! These types are registered in the `savant_rs.nvinfer` Python submodule
//! by `savant_python` when the `deepstream` feature is enabled.

pub(crate) mod batching_operator;
pub(crate) mod config;
pub(crate) mod enums;
pub(crate) mod output;
pub(crate) mod pipeline;
pub(crate) mod roi;

use pyo3::prelude::*;

/// Register all nvinfer Python classes on the given module.
pub fn register_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<enums::PyMetaClearPolicy>()?;
    m.add_class::<enums::PyModelInputScaling>()?;
    m.add_class::<enums::PyModelColorFormat>()?;
    m.add_class::<enums::PyDataType>()?;
    m.add_class::<roi::PyRoi>()?;
    m.add_class::<roi::PyRoiKind>()?;
    m.add_class::<config::PyNvInferConfig>()?;
    m.add_class::<output::PyInferDims>()?;
    m.add_class::<output::PyTensorView>()?;
    m.add_class::<output::PyElementOutput>()?;
    m.add_class::<output::PyBatchInferenceOutput>()?;
    m.add_class::<pipeline::PyNvInferOutput>()?;
    m.add_class::<pipeline::PyNvInfer>()?;
    m.add_class::<batching_operator::PyNvInferBatchingOperatorConfig>()?;
    m.add_class::<batching_operator::PyBatchFormationResult>()?;
    m.add_class::<batching_operator::PyOperatorTensorView>()?;
    m.add_class::<batching_operator::PyOperatorElementOutput>()?;
    m.add_class::<batching_operator::PyOperatorFrameOutput>()?;
    m.add_class::<batching_operator::PySealedDeliveries>()?;
    m.add_class::<batching_operator::PyOperatorOutput>()?;
    m.add_class::<batching_operator::PyOperatorInferenceOutput>()?;
    m.add_class::<batching_operator::PyNvInferBatchingOperator>()?;
    Ok(())
}
