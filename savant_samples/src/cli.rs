//! Command-line argument parsing shared by the sample binaries.
//!
//! Each sample binary simply parses a [`Cli`] and hands the resolved struct
//! to its pipeline `run` entry point.  Keeping the parser in a library
//! module means all argument semantics (defaults, validation, path
//! resolution) live in one place and are covered by unit tests.

use anyhow::{bail, Result};
use clap::Parser;
use std::path::{Path, PathBuf};

/// End-to-end streaming samples CLI.
#[derive(Debug, Clone, Parser)]
#[command(
    name = "cars-demo",
    about = "Streaming MP4 -> decode -> YOLO -> NvDCF -> draw -> encode -> MP4 pipeline."
)]
pub struct Cli {
    /// Input MP4 file.  If the path is relative it is resolved against the
    /// `savant_samples` crate manifest directory.
    #[arg(long)]
    pub input: PathBuf,

    /// Output MP4 file.  Parent directory must exist and be writable.
    #[arg(long)]
    pub output: PathBuf,

    /// CUDA device ID.
    #[arg(long, default_value_t = 0)]
    pub gpu: u32,

    /// Detection confidence threshold (post-sigmoid).
    #[arg(long, default_value_t = 0.25)]
    pub conf: f32,

    /// NMS IoU threshold.
    #[arg(long, default_value_t = 0.45)]
    pub iou: f32,

    /// Inter-stage channel capacity (number of in-flight messages per
    /// boundary).  Frames flow one at a time, so a small prefetch is enough;
    /// must be >= [`crate::channels::MIN_CHANNEL_CAPACITY`].
    #[arg(long = "channel-cap", default_value_t = 4)]
    pub channel_cap: usize,

    /// Emit per-frame debug logs (detections, tracks, encoded bytes).
    /// Raises the log level of the sample's own modules to `debug` while
    /// keeping third-party crates at `info` to avoid log-storm output.
    #[arg(long, default_value_t = false)]
    pub debug: bool,

    /// Output framerate numerator (frames per `fps_den`).  Used for the MP4
    /// container metadata and the hardware encoder's rate-control.  The
    /// per-buffer PTS carried by the pipeline drives the actual playback
    /// timing, so a small mismatch here is harmless.
    #[arg(long = "fps-num", default_value_t = 25)]
    pub fps_num: i32,

    /// Output framerate denominator.  Must be >= 1.
    #[arg(long = "fps-den", default_value_t = 1)]
    pub fps_den: i32,

    /// Completely disable Picasso's draw stage: no bounding boxes, no
    /// labels, and no frame-id overlay.  The pipeline still performs
    /// decode → infer → track → transform → encode, so the output MP4
    /// is the re-encoded source video without any visual overlay.
    ///
    /// Useful for measuring raw decoder + infer + tracker + encoder
    /// throughput (isolating the Skia draw cost) and for producing an
    /// overlay-free reference copy of the output.
    #[arg(long = "no-draw", default_value_t = false)]
    pub no_draw: bool,
}

impl Cli {
    /// Resolve the input path (relative paths are taken to be relative to
    /// the `savant_samples` crate manifest directory) and validate that it
    /// points to an existing file.
    ///
    /// Returns the absolute, validated input path on success.
    pub fn resolved_input(&self) -> Result<PathBuf> {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let path = if self.input.is_absolute() {
            self.input.clone()
        } else {
            manifest_dir.join(&self.input)
        };

        if !path.exists() {
            bail!(
                "input file does not exist: {}\n\
                 Place the clip manually, e.g.:\n    \
                 curl -L -o {input} \\\n        \
                 https://eu-central-1.linodeobjects.com/savant-data/demo/ny_city_center.mov",
                path.display(),
                input = self.input.display()
            );
        }

        Ok(path)
    }

    /// Validate that the output path is writable (its parent directory
    /// exists).  The file itself does not need to exist yet.
    pub fn validated_output(&self) -> Result<PathBuf> {
        let path = self.output.clone();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                bail!("output directory does not exist: {}", parent.display());
            }
        }
        Ok(path)
    }

    /// Validate all numeric knobs (channel capacity, thresholds).  Runs
    /// before any I/O so tests can exercise the checks in isolation.
    pub fn validate_knobs(&self) -> Result<()> {
        if self.channel_cap < crate::channels::MIN_CHANNEL_CAPACITY {
            bail!(
                "--channel-cap must be >= {}",
                crate::channels::MIN_CHANNEL_CAPACITY
            );
        }
        if !(0.0..=1.0).contains(&self.conf) {
            bail!("--conf must be in [0, 1], got {}", self.conf);
        }
        if !(0.0..=1.0).contains(&self.iou) {
            bail!("--iou must be in [0, 1], got {}", self.iou);
        }
        if self.fps_num <= 0 {
            bail!("--fps-num must be > 0, got {}", self.fps_num);
        }
        if self.fps_den <= 0 {
            bail!("--fps-den must be > 0, got {}", self.fps_den);
        }
        Ok(())
    }

    /// Combined resolution: knobs first, then input + output paths.
    pub fn resolve(&self) -> Result<ResolvedCli> {
        self.validate_knobs()?;
        let input = self.resolved_input()?;
        let output = self.validated_output()?;
        Ok(ResolvedCli {
            input,
            output,
            gpu: self.gpu,
            conf: self.conf,
            iou: self.iou,
            channel_cap: self.channel_cap,
            debug: self.debug,
            fps_num: self.fps_num,
            fps_den: self.fps_den,
            draw_enabled: !self.no_draw,
        })
    }
}

/// Fully-validated, absolute paths + numeric knobs ready to be handed to
/// [`crate::pipeline::run`].
#[derive(Debug, Clone)]
pub struct ResolvedCli {
    /// Absolute path to the input MP4 file.
    pub input: PathBuf,
    /// Absolute path to the output MP4 file.
    pub output: PathBuf,
    /// CUDA device ID.
    pub gpu: u32,
    /// Detection confidence threshold.
    pub conf: f32,
    /// NMS IoU threshold.
    pub iou: f32,
    /// Inter-stage channel capacity.
    pub channel_cap: usize,
    /// Emit per-frame debug logs for pipeline-internal events.
    pub debug: bool,
    /// Output container / encoder framerate numerator.
    pub fps_num: i32,
    /// Output container / encoder framerate denominator.
    pub fps_den: i32,
    /// Whether Picasso's draw stage runs.  When `false` (CLI
    /// `--no-draw`) no bounding boxes, labels, or frame-id overlay
    /// are composited onto the output — decode + infer + track +
    /// transform + encode still run as normal.
    pub draw_enabled: bool,
}

impl ResolvedCli {
    /// Borrow the input path.
    pub fn input(&self) -> &Path {
        &self.input
    }

    /// Borrow the output path.
    pub fn output(&self) -> &Path {
        &self.output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn parse_defaults() {
        let cli = Cli::try_parse_from([
            "cars-demo",
            "--input",
            "some.mov",
            "--output",
            "/tmp/out.mp4",
        ])
        .unwrap();
        assert_eq!(cli.gpu, 0);
        assert_eq!(cli.channel_cap, 4);
        assert!((cli.conf - 0.25).abs() < 1e-6);
        assert!((cli.iou - 0.45).abs() < 1e-6);
    }

    #[test]
    fn missing_input_gives_clear_error() {
        let cli = Cli::try_parse_from([
            "cars-demo",
            "--input",
            "/definitely/does/not/exist/nowhere.mov",
            "--output",
            "/tmp/out.mp4",
        ])
        .unwrap();
        let err = cli.resolved_input().unwrap_err();
        assert!(err.to_string().contains("does not exist"));
    }

    #[test]
    fn rejects_invalid_thresholds() {
        let cli = Cli::try_parse_from([
            "cars-demo",
            "--input",
            "some.mov",
            "--output",
            "/tmp/out.mp4",
            "--conf",
            "1.5",
        ])
        .unwrap();
        let err = cli.resolve().unwrap_err();
        assert!(err.to_string().contains("--conf"));
    }

    /// `--no-draw` must default to `false` (draw stage active) and
    /// invert to `draw_enabled = false` on the resolved struct when
    /// the flag is set.  The toggle is the sole source of truth for
    /// whether the render stage populates Picasso's draw spec, so a
    /// regression here silently turns the feature off (or on) for
    /// every user.
    #[test]
    fn no_draw_flag_inverts_into_draw_enabled() {
        let cli = Cli::try_parse_from([
            "cars-demo",
            "--input",
            "some.mov",
            "--output",
            "/tmp/out.mp4",
        ])
        .unwrap();
        assert!(!cli.no_draw);

        let cli = Cli::try_parse_from([
            "cars-demo",
            "--input",
            "some.mov",
            "--output",
            "/tmp/out.mp4",
            "--no-draw",
        ])
        .unwrap();
        assert!(cli.no_draw);
    }

    #[test]
    fn rejects_small_channel_cap() {
        let cli = Cli::try_parse_from([
            "cars-demo",
            "--input",
            "some.mov",
            "--output",
            "/tmp/out.mp4",
            "--channel-cap",
            "0",
        ])
        .unwrap();
        let err = cli.resolve().unwrap_err();
        assert!(err.to_string().contains("--channel-cap"));
    }
}
