//! Minimal GStreamer MP4 muxer: `appsrc -> parser -> qtmux -> filesink`.
//!
//! Accepts raw encoded frames (H.264, HEVC, JPEG, AV1) and writes them into
//! an MP4 (QuickTime) container on disk.

use std::str::FromStr;

use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use thiserror::Error;

use crate::codec::Codec;

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum Mp4MuxerError {
    #[error("Failed to create GStreamer element: {0}")]
    ElementCreation(String),
    #[error("Failed to link elements: {0}")]
    LinkError(String),
    #[error("Pipeline state change failed")]
    StateChangeFailed,
    #[error("Push failed: {0}")]
    PushFailed(String),
    #[error("Muxer already finished")]
    AlreadyFinished,
}

// ---------------------------------------------------------------------------
// Mp4Muxer
// ---------------------------------------------------------------------------

/// Minimal GStreamer pipeline that muxes encoded frames into an MP4 file.
///
/// Pipeline: `appsrc -> parser -> qtmux -> filesink`
pub struct Mp4Muxer {
    pipeline: gst::Pipeline,
    appsrc: gst_app::AppSrc,
    finished: bool,
}

impl Mp4Muxer {
    /// Create and start a new muxer pipeline.
    ///
    /// # Arguments
    ///
    /// * `codec`       – video codec to mux.
    /// * `output_path` – filesystem path for the output `.mp4` file.
    /// * `fps_num`     – framerate numerator.
    /// * `fps_den`     – framerate denominator.
    pub fn new(
        codec: Codec,
        output_path: &str,
        fps_num: i32,
        fps_den: i32,
    ) -> Result<Self, Mp4MuxerError> {
        let _ = gst::init();

        let caps_str = codec.caps_str();
        let parse_name = codec.parser_element();

        let pipeline = gst::Pipeline::new();

        // Create elements
        let appsrc_elem = gst::ElementFactory::make("appsrc")
            .name("mux-src")
            .build()
            .map_err(|e| Mp4MuxerError::ElementCreation(format!("appsrc: {e}")))?;

        let parse = gst::ElementFactory::make(parse_name)
            .name("mux-parse")
            .build()
            .map_err(|e| Mp4MuxerError::ElementCreation(format!("{parse_name}: {e}")))?;

        let mux = gst::ElementFactory::make("qtmux")
            .name("mux")
            .build()
            .map_err(|e| Mp4MuxerError::ElementCreation(format!("qtmux: {e}")))?;

        let sink = gst::ElementFactory::make("filesink")
            .name("mux-sink")
            .build()
            .map_err(|e| Mp4MuxerError::ElementCreation(format!("filesink: {e}")))?;

        // Configure appsrc
        let caps = gst::Caps::from_str(&format!("{caps_str}, framerate={fps_num}/{fps_den}"))
            .map_err(|e| Mp4MuxerError::ElementCreation(format!("caps: {e}")))?;

        appsrc_elem.set_property("caps", &caps);
        appsrc_elem.set_property_from_str("stream-type", "stream");

        sink.set_property("location", output_path);

        // Assemble
        pipeline
            .add_many([&appsrc_elem, &parse, &mux, &sink])
            .map_err(|e| Mp4MuxerError::LinkError(format!("add_many: {e}")))?;

        gst::Element::link_many([&appsrc_elem, &parse, &mux, &sink])
            .map_err(|e| Mp4MuxerError::LinkError(format!("link_many: {e}")))?;

        // Obtain typed AppSrc handle
        let appsrc = appsrc_elem
            .dynamic_cast::<gst_app::AppSrc>()
            .map_err(|_| Mp4MuxerError::ElementCreation("appsrc cast failed".into()))?;

        // Start
        let ret = pipeline.set_state(gst::State::Playing);
        if ret == Err(gst::StateChangeError) {
            return Err(Mp4MuxerError::StateChangeFailed);
        }

        log::info!(
            "Muxer: appsrc -> {} -> qtmux -> filesink({})",
            parse_name,
            output_path,
        );

        Ok(Self {
            pipeline,
            appsrc,
            finished: false,
        })
    }

    /// Push an encoded frame into the muxer pipeline.
    ///
    /// # Arguments
    ///
    /// * `data`        – raw encoded bitstream for a single frame.
    /// * `pts_ns`      – presentation timestamp in nanoseconds.
    /// * `dts_ns`      – optional decode timestamp in nanoseconds.
    ///   Required for streams with B-frames where DTS != PTS.
    /// * `duration_ns` – optional frame duration in nanoseconds.
    pub fn push(
        &mut self,
        data: &[u8],
        pts_ns: u64,
        dts_ns: Option<u64>,
        duration_ns: Option<u64>,
    ) -> Result<(), Mp4MuxerError> {
        if self.finished {
            return Err(Mp4MuxerError::AlreadyFinished);
        }

        let mut buf = gst::Buffer::with_size(data.len())
            .map_err(|e| Mp4MuxerError::PushFailed(format!("buffer alloc: {e}")))?;

        {
            let buf_mut = buf
                .get_mut()
                .ok_or_else(|| Mp4MuxerError::PushFailed("buffer not writable".into()))?;
            buf_mut.set_pts(gst::ClockTime::from_nseconds(pts_ns));
            if let Some(dts) = dts_ns {
                buf_mut.set_dts(gst::ClockTime::from_nseconds(dts));
            }
            if let Some(dur) = duration_ns {
                buf_mut.set_duration(gst::ClockTime::from_nseconds(dur));
            }
            buf_mut
                .copy_from_slice(0, data)
                .map_err(|_| Mp4MuxerError::PushFailed("copy_from_slice failed".into()))?;
        }

        self.appsrc
            .push_buffer(buf)
            .map_err(|e| Mp4MuxerError::PushFailed(format!("{e}")))?;

        Ok(())
    }

    /// Send EOS and shut down the muxer pipeline.
    pub fn finish(&mut self) -> Result<(), Mp4MuxerError> {
        if self.finished {
            return Ok(());
        }
        self.finished = true;

        let _ = self.appsrc.end_of_stream();

        // Wait for EOS or error (5 second timeout)
        if let Some(bus) = self.pipeline.bus() {
            let _msg = bus.timed_pop_filtered(
                gst::ClockTime::from_seconds(5),
                &[gst::MessageType::Eos, gst::MessageType::Error],
            );
        }

        let _ = self.pipeline.set_state(gst::State::Null);
        Ok(())
    }

    /// Whether the muxer has been finalized.
    pub fn is_finished(&self) -> bool {
        self.finished
    }
}

impl Drop for Mp4Muxer {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.finish();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_muxer_lifecycle_h264() {
        gst::init().unwrap();

        let path = "/tmp/test_muxer_lifecycle_h264.mp4";
        let _ = fs::remove_file(path);

        let mut muxer = Mp4Muxer::new(Codec::H264, path, 30, 1).unwrap();

        // Push some minimal H.264 data (SPS/PPS + IDR — just raw bytes for the test;
        // the parser will handle malformed data gracefully enough for a lifecycle test).
        let fake_frame: Vec<u8> = vec![
            0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0x00, 0x0a, 0xe9, 0x40, 0x40, 0x04, 0x00, 0x00,
            0x00, 0x02,
        ];
        for i in 0..3u64 {
            let _ = muxer.push(&fake_frame, i * 33_333_333, None, Some(33_333_333));
        }

        assert!(!muxer.is_finished());
        muxer.finish().unwrap();
        assert!(muxer.is_finished());

        // Double finish is safe
        muxer.finish().unwrap();

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_muxer_lifecycle_hevc() {
        gst::init().unwrap();

        let path = "/tmp/test_muxer_lifecycle_hevc.mp4";
        let _ = fs::remove_file(path);

        let mut muxer = Mp4Muxer::new(Codec::Hevc, path, 30, 1).unwrap();
        assert!(!muxer.is_finished());
        muxer.finish().unwrap();
        assert!(muxer.is_finished());

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_push_after_finish() {
        gst::init().unwrap();

        let path = "/tmp/test_push_after_finish.mp4";
        let _ = fs::remove_file(path);

        let mut muxer = Mp4Muxer::new(Codec::H264, path, 30, 1).unwrap();
        muxer.finish().unwrap();

        let result = muxer.push(&[0u8; 16], 0, None, None);
        assert!(result.is_err());

        let _ = fs::remove_file(path);
    }
}
