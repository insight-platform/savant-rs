//! Detecting-state handler for [`super::FlexibleDecoder`].

use crate::codec_resolve::DetectionStrategy;
use savant_core::primitives::video_codec::VideoCodec;

use super::error::FlexibleDecoderError;
use super::output::{FlexibleDecoderOutput, SkipReason};
use super::state::{
    register_frame, teardown_activated, ActivateFn, BufferedPacket, DecoderState, FrameMap,
    StateGuard, SubmitContext,
};

/// Handle a submit when the decoder is in `Detecting` state.
///
/// On the first random access point (IDR / I-frame) the decoder is
/// activated and the IDR is submitted to it.  Once the IDR is safely
/// accepted, the previously-buffered pre-RAP packets are emitted as
/// [`SkipReason::WaitingForKeyframe`] — they cannot be decoded without the
/// IDR as anchor and replaying them only leaks PTS entries into the
/// underlying GstPipeline `in_flight` map, which later trips the watchdog.
/// If the user wants those frames anyway they can observe them through the
/// `Skipped` callback.
///
/// If activation fails, both the buffered packets and the current one are
/// emitted as `Skipped(DecoderCreationFailed)` and the state is reset to
/// `Idle`.  If activation succeeds but the IDR's first
/// [`NvDecoder::submit_packet`](deepstream_decoders::NvDecoder::submit_packet)
/// errors, the buffered packets are emitted as
/// `Skipped(DecoderCreationFailed)` (they share the same root cause) and
/// the IDR itself is surfaced through the returned `Err`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn handle_detecting(
    guard: StateGuard<'_>,
    pending: &mut Vec<FlexibleDecoderOutput>,
    frame_map: &FrameMap,
    detect_buffer_limit: usize,
    strategy: DetectionStrategy,
    video_codec: VideoCodec,
    width: i64,
    height: i64,
    mut buffered: Vec<BufferedPacket>,
    ctx: &SubmitContext<'_>,
    activate: &ActivateFn<'_>,
) -> Result<(), FlexibleDecoderError> {
    if strategy.is_random_access_point(ctx.payload) {
        if let Some(cfg) = strategy.detect_config(ctx.payload) {
            let real_gst_codec = cfg.codec();
            match activate(cfg, real_gst_codec, width, height, ctx.frame) {
                Ok((decoder, worker_join, worker_stop)) => {
                    // Try the IDR first; only after it is safely accepted
                    // by the decoder do we drain the buffered pre-RAP
                    // packets as `WaitingForKeyframe`.  Draining them up
                    // front would mislabel them on the failure path: if
                    // `submit_packet` errors, the buffered frames were
                    // orphaned by the same decoder failure, not by routine
                    // pre-keyframe skipping.
                    register_frame(frame_map, ctx.frame_id, ctx.frame);
                    match decoder.submit_packet(
                        ctx.payload,
                        ctx.frame_id,
                        ctx.clk.submission_order_ns,
                        ctx.clk.dts_ns,
                        ctx.clk.duration_ns,
                    ) {
                        Ok(()) => {
                            // IDR accepted — pre-RAP packets were truly
                            // skipped while waiting for a keyframe.
                            // Replaying them would strand their PTS in the
                            // GstPipeline watchdog map.
                            for pkt in buffered.drain(..) {
                                pending.push(FlexibleDecoderOutput::Skipped {
                                    frame: pkt.frame,
                                    data: Some(pkt.data),
                                    reason: SkipReason::WaitingForKeyframe,
                                });
                            }
                            guard.commit(DecoderState::Active {
                                decoder,
                                worker_join: Some(worker_join),
                                worker_stop,
                                gst_codec: real_gst_codec,
                                video_codec,
                                width,
                                height,
                            });
                            return Ok(());
                        }
                        Err(e) => {
                            teardown_activated(&decoder, worker_join, &worker_stop);
                            frame_map.lock().remove(&ctx.frame_id);
                            // Surface the buffered frames with the same
                            // reason as the activation-failure branch
                            // below — the freshly-created decoder rejected
                            // its very first packet, so those dependent
                            // pre-RAP frames are lost for the same reason.
                            // The IDR itself is reported via the returned
                            // `Err`, matching the existing contract.
                            let reason_str = format!("{e}");
                            for pkt in buffered.drain(..) {
                                pending.push(FlexibleDecoderOutput::Skipped {
                                    frame: pkt.frame,
                                    data: Some(pkt.data),
                                    reason: SkipReason::DecoderCreationFailed(reason_str.clone()),
                                });
                            }
                            guard.commit(DecoderState::Idle);
                            return Err(e.into());
                        }
                    }
                }
                Err(msg) => {
                    for pkt in buffered.drain(..) {
                        pending.push(FlexibleDecoderOutput::Skipped {
                            frame: pkt.frame,
                            data: Some(pkt.data),
                            reason: SkipReason::DecoderCreationFailed(msg.clone()),
                        });
                    }
                    pending.push(FlexibleDecoderOutput::Skipped {
                        frame: ctx.frame.clone(),
                        data: Some(ctx.payload.to_vec()),
                        reason: SkipReason::DecoderCreationFailed(msg),
                    });
                    guard.commit(DecoderState::Idle);
                    return Ok(());
                }
            }
        }
    }

    buffered.push(BufferedPacket {
        frame: ctx.frame.clone(),
        data: ctx.payload.to_vec(),
    });

    if buffered.len() > detect_buffer_limit {
        for pkt in buffered.drain(..) {
            pending.push(FlexibleDecoderOutput::Skipped {
                frame: pkt.frame,
                data: Some(pkt.data),
                reason: SkipReason::DetectionBufferOverflow,
            });
        }
        guard.commit(DecoderState::Idle);
    } else {
        guard.commit(DecoderState::Detecting {
            strategy,
            video_codec,
            width,
            height,
            buffered,
        });
    }

    Ok(())
}
