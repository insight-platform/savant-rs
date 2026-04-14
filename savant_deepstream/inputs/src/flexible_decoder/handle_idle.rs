//! Idle-state handler for [`super::FlexibleDecoder`].

use crate::codec_resolve::CodecResolve;

use super::error::FlexibleDecoderError;
use super::output::{FlexibleDecoderOutput, SkipReason};
use super::state::{
    register_frame, teardown_activated, ActivateFn, BufferedPacket, DecoderState, FrameMap,
    StateGuard, SubmitContext,
};

/// Handle a submit when the decoder is in `Idle` state.
///
/// On success the guard is committed to `Active` or `Detecting`.
/// On failure the guard is committed to `Idle` and a skip output may be
/// pushed.  Infrastructure errors (e.g. `submit_packet` failure after
/// activation) are returned as `Err` after tearing down the
/// partially-created session.
pub(crate) fn handle_idle(
    guard: StateGuard<'_>,
    pending: &mut Vec<FlexibleDecoderOutput>,
    frame_map: &FrameMap,
    resolve: CodecResolve,
    ctx: &SubmitContext<'_>,
    activate: &ActivateFn<'_>,
) -> Result<(), FlexibleDecoderError> {
    match resolve {
        CodecResolve::Ready(decoder_config) => {
            let gst_codec = decoder_config.codec();
            match activate(decoder_config, gst_codec, ctx.width, ctx.height, ctx.frame) {
                Ok((decoder, worker_join, worker_stop)) => {
                    let vc = match ctx.video_codec {
                        Some(vc) => vc,
                        None => {
                            teardown_activated(&decoder, worker_join, &worker_stop);
                            pending.push(FlexibleDecoderOutput::Skipped {
                                frame: ctx.frame.clone(),
                                data: Some(ctx.payload.to_vec()),
                                reason: SkipReason::UnsupportedCodec(None),
                            });
                            guard.commit(DecoderState::Idle);
                            return Ok(());
                        }
                    };
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
                                gst_codec,
                                video_codec: vc,
                                width: ctx.width,
                                height: ctx.height,
                            });
                            Ok(())
                        }
                        Err(e) => {
                            teardown_activated(&decoder, worker_join, &worker_stop);
                            frame_map.lock().remove(&ctx.frame_id);
                            guard.commit(DecoderState::Idle);
                            Err(e.into())
                        }
                    }
                }
                Err(msg) => {
                    pending.push(FlexibleDecoderOutput::Skipped {
                        frame: ctx.frame.clone(),
                        data: Some(ctx.payload.to_vec()),
                        reason: SkipReason::DecoderCreationFailed(msg),
                    });
                    guard.commit(DecoderState::Idle);
                    Ok(())
                }
            }
        }
        CodecResolve::NeedDetection(strategy) => {
            let vc = match ctx.video_codec {
                Some(vc) => vc,
                None => {
                    pending.push(FlexibleDecoderOutput::Skipped {
                        frame: ctx.frame.clone(),
                        data: Some(ctx.payload.to_vec()),
                        reason: SkipReason::UnsupportedCodec(None),
                    });
                    guard.commit(DecoderState::Idle);
                    return Ok(());
                }
            };
            let mut buffered = Vec::new();

            if strategy.is_random_access_point(ctx.payload) {
                if let Some(cfg) = strategy.detect_config(ctx.payload) {
                    let real_gst_codec = cfg.codec();
                    match activate(cfg, real_gst_codec, ctx.width, ctx.height, ctx.frame) {
                        Ok((decoder, worker_join, worker_stop)) => {
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
                                        video_codec: vc,
                                        width: ctx.width,
                                        height: ctx.height,
                                    });
                                    return Ok(());
                                }
                                Err(e) => {
                                    teardown_activated(&decoder, worker_join, &worker_stop);
                                    frame_map.lock().remove(&ctx.frame_id);
                                    guard.commit(DecoderState::Idle);
                                    return Err(e.into());
                                }
                            }
                        }
                        Err(msg) => {
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
            guard.commit(DecoderState::Detecting {
                strategy,
                video_codec: vc,
                width: ctx.width,
                height: ctx.height,
                buffered,
            });
            Ok(())
        }
    }
}
