//! End-to-end *car-tracking* sample: MP4 -> YOLO11n -> NvDCF -> Picasso -> MP4.
//!
//! The sample is frame-by-frame (no batching: inference and tracking both use
//! `max_batch_size = 1`, `max_batch_wait = 0`) so that latency stays low and
//! memory remains bounded regardless of input length.  Every stage boundary
//! is a bounded [`crossbeam::channel`].
//!
//! # Actor / carrier architecture
//!
//! Each actor module encapsulates one **actor** — its behavior, operator
//! lifecycle, and the message type it emits downstream.  The bounded
//! `crossbeam::channel` between actors is the **carrier** — pure
//! infrastructure.  A given actor is responsible for:
//!
//! 1. Driving its operator (if any): `add_frame` + `send_eos` +
//!    `graceful_shutdown`.
//! 2. Translating upstream messages (deliveries and the in-band
//!    `SourceEos { source_id }` sentinel) into operator calls.
//! 3. Forwarding *its own* downstream message type on its output
//!    channel — including the terminal `SourceEos` after local
//!    drain.
//!
//! [`pipeline`] is the orchestrator: it creates channels, spawns
//! actors, and joins them.  Every actor lives as a submodule of
//! [`pipeline`] (see `cars_tracking/pipeline/<actor>.rs`).
//!
//! # Module layout
//!
//! - [`message`]     – unified ingress message (`PipelineMsg`) shared by
//!   the infer/tracker/picasso stages
//! - [`stats`]       – shared stage counters + sample-level counters
//! - [`supervisor`]  – back-channel + Drop-based `StageExitGuard` used by
//!   the orchestrator to learn about stage exits
//! - [`warmup`]      – engine preparation phase (TensorRT build/load + heartbeat)
//! - [`pipeline`]    – orchestration glue (`run`) + per-actor submodules:
//!   [`pipeline::mp4_demux`], [`pipeline::decoder`],
//!   [`pipeline::infer`] (with [`pipeline::infer::model`] — YOLO11n
//!   nvinfer config + properties), [`pipeline::tracker`],
//!   [`pipeline::picasso`] (with [`pipeline::picasso::draw_spec`]),
//!   [`pipeline::blackhole`] (`--no-picasso` terminus), and
//!   [`pipeline::mp4_mux`].

pub mod message;
pub mod pipeline;
pub mod stats;
pub mod supervisor;
pub mod warmup;
