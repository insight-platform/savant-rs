//! Mux actor for the `cars_tracking` sample.
//!
//! Consumes [`EncodedMsg`] produced by the picasso actor's
//! `OnEncodedFrame` callback and feeds the access units to
//! [`savant_gstreamer::mp4_muxer::Mp4Muxer`].  Picasso is oblivious to
//! what the downstream terminus does with the stream: the muxer
//! decides when to finalise the file based on the in-band
//! `SourceEos { source_id }` sentinel (single-source sample — the
//! first such sentinel is also the last) or on the orchestrator's
//! broadcast [`EncodedMsg::Shutdown`].
//!
//! Fatal push errors surface as `Err(_)` from the thread handle.

use anyhow::{anyhow, Context, Result};
use crossbeam::channel::{Receiver, RecvTimeoutError};
use savant_core::primitives::frame::VideoFrameContent;
use savant_core::primitives::video_codec::VideoCodec;
use savant_gstreamer::mp4_muxer::Mp4Muxer;
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::cars_tracking::message::{apply_shutdown_signal, EncodedMsg};
use crate::cars_tracking::stats::PipelineStats;
use crate::cars_tracking::supervisor::{ExitSender, StageExitGuard, StageName};
// `name` is supplied per spawn so a pipeline with multiple
// concurrent muxers (fan-out) can identify their back-channel exits.

/// Receive poll timeout.  Keeps the loop responsive while there is no
/// encoded traffic so the natural EOS path doesn't have to race
/// through arbitrary buffer bursts.
const MUX_RECV_POLL: Duration = Duration::from_millis(100);

/// Spawn the mux actor.
pub fn spawn_mp4_mux_thread(
    output: String,
    fps_num: i32,
    fps_den: i32,
    rx: Receiver<EncodedMsg>,
    stats: Arc<PipelineStats>,
    exit_tx: ExitSender,
    name: StageName,
) -> Result<JoinHandle<Result<()>>> {
    let guard_name = name.clone();
    thread::Builder::new()
        .name("cars-mp4-mux".into())
        .spawn(move || {
            let _exit_guard = StageExitGuard::new(guard_name, exit_tx);
            mp4_mux_thread(output, fps_num, fps_den, rx, stats, name)
        })
        .context("spawn mp4_mux thread")
}

fn mp4_mux_thread(
    output: String,
    fps_num: i32,
    fps_den: i32,
    rx: Receiver<EncodedMsg>,
    _stats: Arc<PipelineStats>,
    name: StageName,
) -> Result<()> {
    log::info!(
        "[{name}] starting output={output} fps={fps_num}/{fps_den} — acting as pipeline terminus"
    );
    let mut muxer = Mp4Muxer::new(VideoCodec::H264, &output, fps_num, fps_den)
        .map_err(|e| anyhow!("Mp4Muxer::new: {e}"))?;

    // Termination contract:
    //
    // * [`EncodedMsg::Frame`]     — picasso's encoded frame.  Read
    //   timing + payload off the pre-built [`VideoFrameProxy`] and
    //   push onto the muxer.  `payload = Some(bytes)` takes
    //   precedence; `None` means "extract from the frame's
    //   internal content" (matching the decoder-side contract).
    // * [`EncodedMsg::SourceEos`] — in the single-source sample the
    //   first SourceEos is also the last; break and finalise the
    //   `moov` atom.  A multi-source muxer would track in-flight
    //   sources instead.
    // * [`EncodedMsg::Shutdown`]  — cooperative exit broadcast by
    //   the supervisor.  `grace = None` breaks after the current
    //   message is handled; `grace = Some(d)` sets a deadline and
    //   keeps draining pending frames until it expires.
    // * [`EncodedMsg::StreamInfo`] / [`EncodedMsg::Packet`] — not
    //   part of the picasso → mux contract; logged at `debug!` and
    //   ignored so the protocol stays forward-compatible.
    // * `recv_timeout(Disconnected)` — upstream dropped without an
    //   EOS (abnormal); finalise what we have.
    let mut deadline: Option<Instant> = None;
    let mut break_now = false;
    loop {
        match rx.recv_timeout(MUX_RECV_POLL) {
            Ok(EncodedMsg::Frame { frame, payload }) => {
                let pts_ns = frame.get_pts().max(0) as u64;
                let dts_ns = frame.get_dts().map(|v| v.max(0) as u64);
                let duration_ns = frame.get_duration().map(|v| v.max(0) as u64);
                let content = frame.get_content();
                let data: &[u8] = match payload.as_deref() {
                    Some(bytes) => bytes,
                    None => match content.as_ref() {
                        VideoFrameContent::Internal(bytes) => bytes.as_slice(),
                        other => {
                            log::error!(
                                "[{name}] frame source_id={} has no payload and non-Internal content: {:?}",
                                frame.get_source_id(),
                                std::mem::discriminant(other)
                            );
                            continue;
                        }
                    },
                };
                if let Err(e) = muxer.push(data, pts_ns, dts_ns, duration_ns) {
                    log::error!("[{name}] push failed: {e}");
                    return Err(anyhow!("mux push: {e}"));
                }
            }
            Ok(EncodedMsg::SourceEos { source_id: sid }) => {
                log::info!("[{name}] SourceEos {sid}: finalising");
                break;
            }
            Ok(EncodedMsg::Shutdown { grace, reason }) => {
                apply_shutdown_signal(&name, grace, &reason, &mut deadline, &mut break_now);
            }
            Ok(EncodedMsg::StreamInfo { source_id: sid, .. }) => {
                log::debug!("[{name}] ignoring StreamInfo source_id={sid}");
            }
            Ok(EncodedMsg::Packet { source_id: sid, .. }) => {
                log::debug!("[{name}] ignoring Packet source_id={sid}");
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                log::info!("[{name}] upstream channel disconnected without EOS");
                break;
            }
        }
        if break_now {
            break;
        }
        if let Some(d) = deadline {
            if Instant::now() >= d {
                log::info!("[{name}] grace deadline expired; exiting receive loop");
                break;
            }
        }
    }

    muxer
        .finish()
        .map_err(|e| anyhow!("Mp4Muxer::finish: {e}"))?;
    log::info!("[{name}] finished: {}", output);
    Ok(())
}
