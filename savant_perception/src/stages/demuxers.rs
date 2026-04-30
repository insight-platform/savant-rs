//! Demuxer-stage support modules.
//!
//! Houses helper types shared across the demuxer Layer-B stages
//! ([`Mp4DemuxerSource`](super::mp4_demuxer::Mp4DemuxerSource) and
//! [`UriDemuxerSource`](super::uri_demuxer::UriDemuxerSource)) but
//! that are not themselves stages — keeping them out of the
//! flat [`stages`](super) namespace where every other module
//! corresponds to a runnable Layer-B stage.

pub mod decode_frame;
pub mod demux_input;

pub use decode_frame::{make_decode_frame, FALLBACK_FPS_DEN, FALLBACK_FPS_NUM};
pub use demux_input::{looped_requester, one_shot_requester, DemuxInputRequest, InputRequester};
