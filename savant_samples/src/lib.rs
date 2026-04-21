//! End-to-end streaming video-analytics samples built on savant-rs.
//!
//! This crate hosts thin, readable end-to-end pipelines that exercise the
//! building blocks from [`deepstream_inputs`], [`deepstream_nvinfer`],
//! [`deepstream_nvtracker`], [`picasso`] and [`savant_gstreamer`] in a single
//! place.  Each sample is intentionally structured as a set of small modules
//! (one per pipeline stage) wired together via bounded
//! [`crossbeam::channel`]s so that memory usage stays bounded regardless of
//! input duration.
//!
//! # Streaming invariants
//!
//! Every sample in this crate obeys the following rules:
//!
//! 1. **No full-file materialization.**  The demuxer uses a push callback
//!    ([`Mp4Demuxer::new_parsed`](savant_gstreamer::Mp4Demuxer::new_parsed)),
//!    the decoder is backed by a fixed-size NVMM pool, and the muxer writes
//!    each encoded access unit directly to `filesink`.
//! 2. **Backpressure via bounded channels.**  All stage boundaries use
//!    [`crossbeam::channel::bounded`] with a small capacity.  A slow
//!    downstream stage blocks its upstream producer.
//! 3. **EOS by sender drop.**  Shutdown is driven by the orchestrator
//!    dropping upstream senders; each stage calls `graceful_shutdown()` on
//!    the operator it owns once its input channel closes.  Operator-level
//!    source-EOS notifications are only logged; the muxer flushes the
//!    `moov` atom after receiving an [`EncodedMsg::Eos`](crate::channels::EncodedMsg)
//!    sentinel from the Picasso encoder thread.
//! 4. **Per-frame ownership.**  Each frame traverses the pipeline as a single
//!    `(VideoFrameProxy, SharedBuffer)` tuple packaged into a sealed
//!    delivery; [`SharedBuffer`] clones are dropped as soon as the next
//!    stage has consumed them so NVMM slots return to the pool.
//! 5. **Unseal outside callbacks.**  Operator result callbacks never call
//!    `unseal()`; they forward the `SealedDeliveries` through a bounded
//!    channel so a dedicated consumer thread unseals and submits to the
//!    next stage, preserving backpressure onto the operator's internal
//!    completion thread.
//!
//! # Available samples
//!
//! | Binary | Description |
//! |--------|-------------|
//! | `cars-demo` | Detect + track + draw vehicles with YOLO11n + NvDCF, save to MP4. |

pub mod assets;
pub mod cars_tracking;
pub mod channels;
pub mod cli;
