//! Inference stage helpers for the `cars_demo` sample.
//!
//! Responsibilities are split across this module and two submodules:
//!
//! * [`model`] — nvinfer engine/config, label tables, YOLO class
//!   constants.
//! * [`output`] — concrete tensor-output processing: the
//!   [`process_infer_output`](output::process_infer_output) dispatcher
//!   used by the framework actor, plus the
//!   [`InferStats`](output::InferStats) counters.
//! * *This module* — the batch-formation callback handed to
//!   [`NvInferBatchingOperator`](deepstream_nvinfer::prelude::NvInferBatchingOperator).
//!
//! The operator's completion thread is the single vantage point where
//! "no more deliveries for this source will follow" is an invariant;
//! that's why [`output::process_infer_output`] is the only place that
//! emits the in-band [`PipelineMsg::SourceEos`] sentinel onto the
//! infer → tracker channel.  See the module-level docs on
//! [`output`] for the full stream-alignment argument.
//!
//! [`PipelineMsg::SourceEos`]: savant_perception::envelopes::PipelineMsg::SourceEos

use deepstream_buffers::SavantIdMetaKind;
use deepstream_nvinfer::{BatchFormationCallback, BatchFormationResult, RoiKind};
use std::sync::Arc;

pub mod model;
pub mod output;

/// Build the batch-formation callback used by
/// [`NvInferBatchingOperator`](deepstream_nvinfer::prelude::NvInferBatchingOperator).
///
/// Each slot processes a single full-frame ROI (the whole image) —
/// inference runs at `max_batch_size = 1` in this sample.
pub fn build_batch_formation() -> BatchFormationCallback {
    Arc::new(|frames| {
        let ids = frames
            .iter()
            .enumerate()
            .map(|(slot, _)| SavantIdMetaKind::Frame(slot as u128))
            .collect();
        let rois = frames.iter().map(|_| RoiKind::FullFrame).collect();
        BatchFormationResult { ids, rois }
    })
}
