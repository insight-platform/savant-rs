//! End-to-end *cars-demo* sample: MP4 → YOLO11n → NvDCF → Picasso → MP4.
//!
//! The sample is frame-by-frame (no batching: inference and tracking
//! both use `max_batch_size = 1`, `max_batch_wait = 0`) so that
//! latency stays low and memory remains bounded regardless of input
//! length.  Every stage boundary is a bounded [`crossbeam::channel`].
//!
//! # Architecture
//!
//! The pipeline is a [`savant_perception`](crate) actor system.  The
//! orchestrator in [`pipeline`] registers one source (the MP4
//! demuxer), one actor per stage (decoder, infer, tracker, either
//! Picasso followed by an MP4 muxer or a blackhole terminus), and
//! delegates thread spawning, channel allocation, address
//! publication, cooperative shutdown, and join to
//! [`savant_perception::System`].
//!
//! Sample-specific logic stays in sibling submodules as small,
//! reusable helpers:
//!
//! * [`pipeline::infer::model`] — YOLO11n nvinfer config + class
//!   tables.
//! * [`pipeline::infer::output`] — YOLO tensor → detections
//!   conversion, `process_infer_output` dispatcher, and
//!   [`InferStats`](pipeline::infer::output::InferStats).
//! * [`pipeline::tracker`] — NvDCF config + batch formation +
//!   `process_tracker_output` reconciler + [`TrackerStats`](pipeline::tracker::TrackerStats).
//! * [`pipeline::picasso`] + [`pipeline::picasso::draw_spec`] —
//!   per-source `SourceSpec` builder + vehicle draw spec + frame-id
//!   overlay.
//!
//! # Module layout
//!
//! - [`stats`]      — shared stage counters + sample-level
//!   [`stats::PipelineStats`] (demux packets / encoded bytes).
//! - [`warmup`]     — TensorRT/NvDCF engine preparation with
//!   progress heartbeats.
//! - [`pipeline`]   — orchestrator (`run`) + per-stage helper
//!   submodules.
//!
//! The unified ingress envelopes ([`PipelineMsg`](savant_perception::envelopes::PipelineMsg)
//! and [`EncodedMsg`](savant_perception::envelopes::EncodedMsg)) plus their
//! [`Envelope`](savant_perception::Envelope) /
//! [`Dispatch`](savant_perception::Dispatch) integration now live in
//! [`savant_perception::envelopes`].
//!
//! Back-channel / guard / stage-name types live under
//! [`savant_perception::supervisor`] — this sample imports them
//! directly rather than re-exporting.

pub mod pipeline;
pub mod stats;
pub mod warmup;
