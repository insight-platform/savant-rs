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
/// On activation, replays buffered packets. If any replay submission fails,
/// the entire activation is aborted: the decoder is torn down, all buffered
/// packets (plus the current one) are emitted as `Skipped`, and the guard
/// is committed to `Idle`.
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
                    let mut registered_ids: Vec<u128> = Vec::new();
                    let mut replay_err: Option<String> = None;

                    for pkt in &buffered {
                        register_frame(frame_map, pkt.frame_id, &pkt.frame);
                        registered_ids.push(pkt.frame_id);
                        if let Err(e) = decoder.submit_packet(
                            &pkt.data,
                            pkt.frame_id,
                            pkt.pts_ns,
                            pkt.dts_ns,
                            pkt.duration_ns,
                        ) {
                            replay_err = Some(format!("buffered packet replay failed: {e}"));
                            break;
                        }
                    }

                    if let Some(err_msg) = replay_err {
                        teardown_activated(&decoder, worker_join, &worker_stop);
                        {
                            let mut fm = frame_map.lock();
                            for id in &registered_ids {
                                fm.remove(id);
                            }
                        }
                        for pkt in buffered {
                            pending.push(FlexibleDecoderOutput::Skipped {
                                frame: pkt.frame,
                                data: Some(pkt.data),
                                reason: SkipReason::DecoderCreationFailed(err_msg.clone()),
                            });
                        }
                        pending.push(FlexibleDecoderOutput::Skipped {
                            frame: ctx.frame.clone(),
                            data: Some(ctx.payload.to_vec()),
                            reason: SkipReason::DecoderCreationFailed(err_msg),
                        });
                        guard.commit(DecoderState::Idle);
                        return Ok(());
                    }

                    register_frame(frame_map, ctx.frame_id, ctx.frame);
                    match decoder.submit_packet(
                        ctx.payload,
                        ctx.frame_id,
                        ctx.clk.submission_order_ns,
                        ctx.clk.dts_ns,
                        ctx.clk.duration_ns,
                    ) {
                        Ok(()) => {
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
                            {
                                let mut fm = frame_map.lock();
                                for id in &registered_ids {
                                    fm.remove(id);
                                }
                                fm.remove(&ctx.frame_id);
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
        frame_id: ctx.frame_id,
        data: ctx.payload.to_vec(),
        pts_ns: ctx.clk.submission_order_ns,
        dts_ns: ctx.clk.dts_ns,
        duration_ns: ctx.clk.duration_ns,
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
