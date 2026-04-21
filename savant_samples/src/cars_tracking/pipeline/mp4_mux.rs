//! Mux actor for the `cars_tracking` sample.
//!
//! Sink of the pipeline when Picasso is enabled: consumes
//! [`EncodedMsg`] from the picasso actor and feeds the access units to
//! [`savant_gstreamer::mp4_muxer::Mp4Muxer`], flushing the `moov` atom
//! on [`EncodedMsg::Eos`].
//!
//! The muxer is the **terminus** of the Picasso pipeline.  On
//! [`EncodedMsg::Eos`] it finalises the file and exits; the
//! orchestrator joins this thread first and then broadcasts
//! `Shutdown` to every upstream stage.  Fatal push errors surface
//! as `Err(_)`.

use anyhow::{anyhow, Context, Result};
use crossbeam::channel::{Receiver, RecvTimeoutError};
use savant_core::primitives::video_codec::VideoCodec;
use savant_gstreamer::mp4_muxer::Mp4Muxer;
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use super::picasso::EncodedMsg;
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
    thread::Builder::new()
        .name("cars-mp4-mux".into())
        .spawn(move || {
            let _exit_guard = StageExitGuard::new(name, exit_tx);
            mp4_mux_thread(output, fps_num, fps_den, rx, stats)
        })
        .context("spawn mp4_mux thread")
}

fn mp4_mux_thread(
    output: String,
    fps_num: i32,
    fps_den: i32,
    rx: Receiver<EncodedMsg>,
    _stats: Arc<PipelineStats>,
) -> Result<()> {
    log::info!(
        "[mp4_mux] starting output={output} fps={fps_num}/{fps_den} — acting as pipeline terminus"
    );
    let mut muxer = Mp4Muxer::new(VideoCodec::H264, &output, fps_num, fps_den)
        .map_err(|e| anyhow!("Mp4Muxer::new: {e}"))?;

    // Termination contract:
    //
    // * [`EncodedMsg::Eos`] — fired by Picasso's `OnEncodedFrame`
    //   callback on `OutputMessage::EndOfStream` (stream-aligned
    //   with the last access unit).  By the time it lands every
    //   frame has traversed the pipeline end-to-end; break and let
    //   the orchestrator broadcast `Shutdown` to every upstream
    //   stage.
    // * On `muxer.push` error return `Err(_)` — the orchestrator
    //   surfaces the failure after joining.
    // * `recv_timeout(Disconnected)` — upstream dropped its sender
    //   before emitting `Eos` (e.g. fatal upstream happened); we
    //   still finalise the muxer on what we have.
    loop {
        match rx.recv_timeout(MUX_RECV_POLL) {
            Ok(EncodedMsg::AccessUnit {
                data,
                pts_ns,
                dts_ns,
                duration_ns,
            }) => {
                if let Err(e) = muxer.push(&data, pts_ns, dts_ns, duration_ns) {
                    log::error!("[mp4_mux] push failed: {e}");
                    return Err(anyhow!("mux push: {e}"));
                }
            }
            Ok(EncodedMsg::Eos) => {
                log::info!("[mp4_mux] EOS: finalising");
                break;
            }
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => {
                log::info!("[mp4_mux] upstream channel disconnected without EOS");
                break;
            }
        }
    }

    muxer
        .finish()
        .map_err(|e| anyhow!("Mp4Muxer::finish: {e}"))?;
    log::info!("[mp4_mux] finished: {}", output);
    Ok(())
}
