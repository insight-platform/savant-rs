//! PyO3 bindings for the `nvtracker` crate.

pub(crate) mod batching_operator;
pub(crate) mod config;
pub(crate) mod enums;
pub(crate) mod output;
pub(crate) mod pipeline;

use pyo3::prelude::*;

/// Register all nvtracker Python classes on the given module.
pub fn register_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<enums::PyTrackingIdResetMode>()?;
    m.add_class::<enums::PyTrackState>()?;
    // `MetaClearPolicy` is registered in the shared `savant_rs.deepstream`
    // submodule; nvtracker imports it from there.
    m.add_class::<config::PyNvTrackerConfig>()?;
    m.add_class::<output::PyTrackedObject>()?;
    m.add_class::<output::PyMiscTrackFrame>()?;
    m.add_class::<output::PyMiscTrackData>()?;
    m.add_class::<output::PyTrackerOutput>()?;
    m.add_class::<batching_operator::PyNvTrackerBatchingOperatorConfig>()?;
    m.add_class::<batching_operator::PyTrackerBatchFormationResult>()?;
    m.add_class::<batching_operator::PyTrackerOperatorFrameOutput>()?;
    m.add_class::<batching_operator::PySealedDeliveries>()?;
    m.add_class::<batching_operator::PyTrackerOperatorOutput>()?;
    m.add_class::<batching_operator::PyNvTrackerBatchingOperator>()?;
    m.add_class::<pipeline::PyTrackedFrame>()?;
    m.add_class::<pipeline::PyNvTrackerOutput>()?;
    m.add_class::<pipeline::PyNvTracker>()?;
    Ok(())
}
