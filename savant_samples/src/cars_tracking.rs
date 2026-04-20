//! End-to-end *car-tracking* sample: MP4 -> YOLO11n -> NvDCF -> Picasso -> MP4.
//!
//! The sample is frame-by-frame (no batching: inference and tracking both use
//! `max_batch_size = 1`, `max_batch_wait = 0`) so that latency stays low and
//! memory remains bounded regardless of input length.  Every stage boundary
//! is a bounded [`crossbeam::channel`], every inference/tracking result is
//! unsealed by a dedicated consumer thread (never inside the operator
//! callback), and shutdown is driven by the orchestrator dropping senders —
//! operator-level source-EOS notifications are only logged.
//!
//! Module layout:
//!
//! - [`model`]       – YOLO11n nvinfer config + properties
//! - [`detections`]  – YOLO tensor post-processing + result callback
//! - [`tracker`]     – NvDCF tracker config + result callback
//! - [`draw`]        – Picasso per-class draw specification
//! - [`pipeline`]    – orchestration glue (`run`)

pub mod detections;
pub mod draw;
pub mod model;
pub mod pipeline;
pub mod tracker;
