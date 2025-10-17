use anyhow::{bail, Result};
use ffmpeg_next as ffmpeg;
use log::{error, info};

// const PARAMS: [(&str, &str); 2] = [("rtsp_transport", "tcp"), ("stream_loop", "-1")];
//const PARAMS: [(&str, &str); 1] = [("rtsp_transport", "tcp")];
const PARAMS: [(&str, &str); 1] = [("stream_loop", "-1")];


ffmpeg::init()?;

let mut opts = ffmpeg::Dictionary::new();
for (k, v) in &PARAMS {
    opts.set(k, v);
}

let mut context = ffmpeg::format::input_with_dictionary(&stream_name, opts)?;

let video_stream_id;
let time_base;
{
    let video_input = match context.streams().best(ffmpeg_next::media::Type::Video) {
        Some(s) => s,
        None => {
            let msg = "Cannot discover the best suitable video stream to work with.";
            error!("{}", msg);
            bail!(msg);
        }
    };
    video_stream_id = video_input.id();
    time_base = video_input.time_base();
    info!("Time base: {:?}", time_base);
    let codec = ffmpeg::codec::context::Context::from_parameters(video_input.parameters())?;
    info!("Codec: {:?}", codec.id());
    if codec.medium() == ffmpeg::media::Type::Video {
        if let Ok(video) = codec.decoder().video() {
            info!("\tbit_rate: {}", video.bit_rate());
            info!("\tmax_rate: {}", video.max_bit_rate());
            info!("\tdelay: {}", video.delay());
            info!("\tvideo.width: {}", video.width());
            info!("\tvideo.height: {}", video.height());
            info!("\tvideo.format: {:?}", video.format());
            info!("\tvideo.has_b_frames: {}", video.has_b_frames());
            info!("\tvideo.aspect_ratio: {}", video.aspect_ratio());
            info!("\tvideo.color_space: {:?}", video.color_space());
            info!("\tvideo.color_range: {:?}", video.color_range());
            info!("\tvideo.color_primaries: {:?}", video.color_primaries());
            info!(
                "\tvideo.color_transfer_characteristic: {:?}",
                video.color_transfer_characteristic()
            );
            info!("\tvideo.chroma_location: {:?}", video.chroma_location());
            info!("\tvideo.references: {}", video.references());
            info!("\tvideo.intra_dc_precision: {}", video.intra_dc_precision());
        }
    }
}
for (stream, packet) in context.packets() {
    if stream.id() == video_stream_id {
        let data_ref = packet.data().unwrap();
        info!(
            "Stream: {}, Packet tag: {:?}, Packet data: {}, is_key_frame: {}, pts: {}, dts: {}, duration: {}, time_base: {}",
            stream.id(),
            &data_ref[0..4],
            data_ref.len(),
            packet.is_key(),
            packet.pts().unwrap_or(0),
            packet.dts().unwrap_or(0),
            packet.duration(),
            time_base
        );
    }
}
