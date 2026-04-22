//! Picasso-engine helpers for the `cars_tracking` sample.
//!
//! The [`Picasso`](crate::framework::templates::Picasso)
//! owns the receive loop, source-spec registration, and
//! [`PicassoEngine`](picasso::prelude::PicassoEngine) lifecycle; this
//! module only exports the sample-specific configuration the actor
//! factories plug in:
//!
//! * [`build_source_spec`] — per-source
//!   [`SourceSpec`](picasso::prelude::SourceSpec) used the first time
//!   a given `source_id` shows up.  Carries the codec + encoder
//!   configuration, the vehicle draw spec (or an empty one under
//!   `--no-draw`), and fixed font/`use_on_*` settings.
//! * [`draw_spec`] — the vehicle-specific
//!   [`ObjectDrawSpec`](picasso::prelude::ObjectDrawSpec) + the
//!   `frame #N` overlay attached immediately before each frame is
//!   handed to the engine.

pub mod draw_spec;

use anyhow::{Context, Result};
use deepstream_encoders::{EncoderConfig, H264EncoderConfig, NvEncoderConfig};
use deepstream_nvinfer::prelude::VideoFormat as InferVideoFormat;
use picasso::prelude::{CodecSpec, ObjectDrawSpec, SourceSpec, TransformConfig};

#[cfg(not(target_arch = "aarch64"))]
use deepstream_encoders::properties::H264DgpuProps;
#[cfg(target_arch = "aarch64")]
use deepstream_encoders::properties::H264JetsonProps;

use self::draw_spec::build_vehicle_draw_spec;

/// Build a per-source [`SourceSpec`] used to register a new
/// `source_id` with the Picasso engine.
///
/// `draw_enabled = false` is the `--no-draw` escape hatch: the
/// returned [`ObjectDrawSpec`] is empty, so Picasso composites no
/// overlays, but the transform + encode stages still run so the
/// output MP4 is a clean re-encoded copy of the source.
pub fn build_source_spec(
    width: u32,
    height: u32,
    fps_num: i32,
    fps_den: i32,
    draw_enabled: bool,
) -> Result<SourceSpec> {
    let encoder = build_encoder_config(width, height, fps_num, fps_den);
    let draw = if draw_enabled {
        build_vehicle_draw_spec().context("build vehicle draw spec")?
    } else {
        ObjectDrawSpec::default()
    };
    Ok(SourceSpec {
        codec: CodecSpec::Encode {
            transform: TransformConfig::default(),
            encoder: Box::new(encoder),
        },
        draw,
        font_family: "monospace".to_string(),
        use_on_render: false,
        use_on_gpumat: false,
        ..Default::default()
    })
}

fn build_encoder_config(width: u32, height: u32, fps_num: i32, fps_den: i32) -> NvEncoderConfig {
    let cfg = H264EncoderConfig::new(width, height)
        .format(InferVideoFormat::RGBA)
        .fps(fps_num, fps_den);
    #[cfg(target_arch = "aarch64")]
    let cfg = cfg.props(H264JetsonProps {
        bitrate: Some(6_000_000),
        iframeinterval: Some(fps_num.max(1) as u32),
        ..Default::default()
    });
    #[cfg(not(target_arch = "aarch64"))]
    let cfg = cfg.props(H264DgpuProps {
        bitrate: Some(6_000_000),
        iframeinterval: Some(fps_num.max(1) as u32),
        ..Default::default()
    });
    NvEncoderConfig::new(0, EncoderConfig::H264(cfg)).name("cars-demo/enc")
}
