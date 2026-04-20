//! Generic, shared channel primitives used by the sample pipelines.
//!
//! Sample-specific messages (e.g. sealed inference/tracker deliveries) live in
//! their respective modules — this file only hosts the messages that cross
//! the *sample-agnostic* demux and mux boundaries:
//!
//! - [`DemuxMsg`] — demuxer -> decoder (one [`DemuxedPacket`] per element,
//!   prefaced by one [`DemuxMsg::StreamInfo`] carrying the container's
//!   [`VideoInfo`] before any packet).
//! - [`EncodedMsg`] — encoder -> muxer (one encoded access unit per element).
//!
//! Every stage boundary uses [`crossbeam::channel::bounded`] so a slow
//! downstream stage blocks its upstream producer; that is how memory stays
//! bounded for arbitrarily long inputs.

use crossbeam::channel::{bounded, Receiver, Sender};
use savant_gstreamer::mp4_demuxer::{DemuxedPacket, VideoInfo};

/// Minimum channel capacity accepted by the helpers below.
///
/// Since every stage processes frames one at a time (`max_batch_size = 1`),
/// we only need a tiny amount of prefetch to hide stage-to-stage scheduling
/// jitter.  Two slots is the smallest capacity that lets one message sit in
/// flight while the producer prepares the next.
pub const MIN_CHANNEL_CAPACITY: usize = 2;

/// Demux -> decode boundary.
///
/// The demuxer always emits exactly one [`DemuxMsg::StreamInfo`] before the
/// first [`DemuxMsg::Packet`]; the decode stage relies on this ordering to
/// size its [`VideoFrameProxy`] metadata (and therefore the NVDEC/RGBA buffer
/// pools) from the container's real width/height/framerate.
#[derive(Debug)]
pub enum DemuxMsg {
    /// Stream metadata discovered from the container caps.  Fires once,
    /// before any [`DemuxMsg::Packet`].
    StreamInfo(VideoInfo),
    /// A single demuxed access unit ready for the decoder.
    Packet(DemuxedPacket),
    /// End of the input file.
    Eos,
    /// A fatal demuxer error; stages treat this as EOS and exit.
    Error(String),
}

/// Encoded-frame message delivered from the Picasso engine's
/// `on_encoded_frame` callback to the muxer thread.
#[derive(Debug)]
pub enum EncodedMsg {
    /// One encoded access unit.
    AccessUnit {
        /// Encoded bitstream bytes (H.264 Annex-B access unit).
        data: Vec<u8>,
        /// Presentation timestamp in nanoseconds.
        pts_ns: u64,
        /// Decode timestamp in nanoseconds, if available.
        dts_ns: Option<u64>,
        /// Frame duration in nanoseconds, if available.
        duration_ns: Option<u64>,
    },
    /// End of the encoded stream; the muxer calls `finish()` and exits.
    Eos,
}

/// Alias for the demux -> decode channel sender.
pub type DemuxSender = Sender<DemuxMsg>;
/// Alias for the demux -> decode channel receiver.
pub type DemuxReceiver = Receiver<DemuxMsg>;

/// Alias for the render -> mux channel sender.
pub type EncodedSender = Sender<EncodedMsg>;
/// Alias for the render -> mux channel receiver.
pub type EncodedReceiver = Receiver<EncodedMsg>;

/// Create a bounded demuxer channel.  `cap` must be at least
/// [`MIN_CHANNEL_CAPACITY`] (checked in debug builds).
pub fn demux_channel(cap: usize) -> (DemuxSender, DemuxReceiver) {
    debug_assert!(cap >= MIN_CHANNEL_CAPACITY);
    bounded(cap)
}

/// Create a bounded encoder -> muxer channel.
pub fn encoded_channel(cap: usize) -> (EncodedSender, EncodedReceiver) {
    debug_assert!(cap >= MIN_CHANNEL_CAPACITY);
    bounded(cap)
}
