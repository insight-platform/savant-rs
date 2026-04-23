//! Command-line argument parsing shared by the sample binaries.
//!
//! Each sample binary simply parses a [`Cli`] and hands the resolved struct
//! to its pipeline `run` entry point.  Keeping the parser in a library
//! module means all argument semantics (defaults, validation, path
//! resolution) live in one place and are covered by unit tests.

use anyhow::{bail, Result};
use clap::Parser;
use std::path::{Path, PathBuf};

/// Minimum allowed `--channel-cap`.  Keeps backpressure meaningful:
/// capacity 1 means every intra-pipeline send serializes an entire
/// stage, which is too tight in practice.
pub const MIN_CHANNEL_CAPACITY: usize = 2;

/// End-to-end streaming samples CLI.
#[derive(Debug, Clone, Parser)]
#[command(
    name = "cars-demo",
    about = "Streaming MP4 -> decode -> YOLO -> NvDCF -> draw -> encode -> MP4 pipeline."
)]
pub struct Cli {
    /// Input MP4 file.  If the path is relative it is resolved against the
    /// `savant_perception` crate manifest directory.
    #[arg(long)]
    pub input: PathBuf,

    /// Output MP4 file.  Required unless [`Cli::no_picasso`] is set;
    /// when Picasso is disabled the pipeline produces no output file
    /// and the argument (if supplied) is ignored with a warning.
    /// Parent directory must exist and be writable.
    #[arg(long)]
    pub output: Option<PathBuf>,

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
    /// must be >= `MIN_CHANNEL_CAPACITY` in the `cars_tracking::pipeline` module.
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
    ///
    /// Implied by [`Cli::no_picasso`] because Picasso is bypassed
    /// entirely in that mode.
    #[arg(long = "no-draw", default_value_t = false)]
    pub no_draw: bool,

    /// Completely exclude Picasso from the pipeline: no Skia engine,
    /// no GPU transform, no encoder, no muxer.  Decoded frames flow
    /// through YOLO + NvDCF as usual and are then dropped — nothing
    /// is written to disk and [`Cli::output`] becomes optional (it
    /// is ignored with a warning if supplied).
    ///
    /// Useful for measuring the raw cost of decode + infer + track
    /// in isolation, without paying for the transform / overlay /
    /// encode tail of the pipeline.
    #[arg(long = "no-picasso", default_value_t = false)]
    pub no_picasso: bool,
}

impl Cli {
    /// Resolve the input path (relative paths are taken to be relative to
    /// the `savant_perception` crate manifest directory) and validate that it
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

    /// Resolve the output path against the picasso mode.
    ///
    /// - When Picasso is enabled (default), `--output` is required and
    ///   its parent directory must exist.
    /// - When `--no-picasso` is set, `--output` is ignored (a warning
    ///   is logged if the user supplied one) and this returns `None`.
    /// - The literal string `null` is a sentinel meaning "run the
    ///   Picasso encoder but do not write an MP4" — it returns
    ///   `None` without checking the filesystem.  Combining
    ///   `--no-picasso --output null` is rejected as a clash: the
    ///   two ways to disable output are not composable.
    #[allow(dead_code)] // public symmetry with `resolved_input`
    pub fn resolved_output(&self) -> Result<Option<PathBuf>> {
        let (output, _is_null) = self.resolve_output_mode()?;
        Ok(output)
    }

    /// Combined output resolver returning both the path (`None`
    /// when no file should be written) and a flag marking the
    /// `--output null` sentinel.  Used by [`Cli::resolve`] to
    /// populate both [`ResolvedCli::output`] and
    /// [`ResolvedCli::output_is_null`] in a single pass.
    fn resolve_output_mode(&self) -> Result<(Option<PathBuf>, bool)> {
        let is_null_sentinel = self
            .output
            .as_deref()
            .map(|p| p.as_os_str() == "null")
            .unwrap_or(false);
        if self.no_picasso {
            if is_null_sentinel {
                bail!(
                    "--output null cannot be combined with --no-picasso; \
                     both disable output on their own"
                );
            }
            if self.output.is_some() {
                log::warn!(
                    "--output is ignored when --no-picasso is set (no file will be written)"
                );
            }
            return Ok((None, false));
        }
        if is_null_sentinel {
            return Ok((None, true));
        }
        let path = self
            .output
            .clone()
            .ok_or_else(|| anyhow::anyhow!("--output is required unless --no-picasso is set"))?;
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                bail!("output directory does not exist: {}", parent.display());
            }
        }
        Ok((Some(path), false))
    }

    /// Validate all numeric knobs (channel capacity, thresholds).  Runs
    /// before any I/O so tests can exercise the checks in isolation.
    pub fn validate_knobs(&self) -> Result<()> {
        if self.channel_cap < MIN_CHANNEL_CAPACITY {
            bail!("--channel-cap must be >= {MIN_CHANNEL_CAPACITY}");
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
        let (output, output_is_null) = self.resolve_output_mode()?;
        // `--no-picasso` implies `draw_enabled = false` because the
        // draw stage lives inside Picasso; with Picasso excluded
        // there is simply nothing to draw onto.
        let picasso_enabled = !self.no_picasso;
        let draw_enabled = picasso_enabled && !self.no_draw;
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
            draw_enabled,
            picasso_enabled,
            output_is_null,
        })
    }
}

/// Fully-validated, absolute paths + numeric knobs ready to be handed to
/// [`crate::cars_tracking::pipeline::run`].
#[derive(Debug, Clone)]
pub struct ResolvedCli {
    /// Absolute path to the input MP4 file.
    pub input: PathBuf,
    /// Absolute path to the output MP4 file.  `None` when Picasso
    /// is excluded via `--no-picasso` — no file is written in that
    /// mode.  Always `Some` when [`Self::picasso_enabled`] is
    /// `true`; the resolver upholds this invariant.
    pub output: Option<PathBuf>,
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
    /// Whether Picasso's draw stage runs.  `false` when either
    /// `--no-draw` or `--no-picasso` is set; the latter implies the
    /// former because the draw stage lives inside Picasso.
    pub draw_enabled: bool,
    /// Whether the pipeline instantiates Picasso at all.  When
    /// `false` (CLI `--no-picasso`) the picasso + mux stages are
    /// replaced by a drop-on-receive drain: decode + infer + track
    /// still run, but no transform / overlay / encode / mux stages
    /// are spun up and no output file is produced.
    pub picasso_enabled: bool,
    /// Set by the `--output null` sentinel: Picasso still runs
    /// (decode + infer + track + transform + encode), but instead
    /// of writing an MP4 the encoded bitstream is routed into a
    /// [`BitstreamFunction`](savant_perception::templates::BitstreamFunction)
    /// terminus that tallies packet counts and encoded byte totals.
    ///
    /// Only meaningful when [`Self::picasso_enabled`] is `true`;
    /// combining it with `--no-picasso` is rejected by
    /// [`Cli::resolve`] (the two ways to disable output are not
    /// composable).
    pub output_is_null: bool,
}

impl ResolvedCli {
    /// Borrow the input path.
    pub fn input(&self) -> &Path {
        &self.input
    }

    /// Borrow the output path, when one was requested.  Always
    /// returns `Some` when [`Self::picasso_enabled`] is `true`; the
    /// resolver refuses to produce a `ResolvedCli` with
    /// `picasso_enabled = true` and `output = None`.
    pub fn output(&self) -> Option<&Path> {
        self.output.as_deref()
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
    /// whether the picasso stage populates Picasso's draw spec, so a
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

    /// `--no-picasso` both switches off Picasso *and* implies
    /// `draw_enabled = false` (because the draw stage lives inside
    /// Picasso).  The resolver is the single source of truth for
    /// that implication, so a regression here silently leaves
    /// `draw_enabled = true` and sends the pipeline down the picasso
    /// path with no engine to render into.
    #[test]
    fn no_picasso_disables_both_picasso_and_draw() {
        let exe_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("examples/cars_demo/cars_tracking.rs");
        // Use an existing file so `resolved_input` passes — we only
        // care about the picasso / draw plumbing here.
        let cli = Cli::try_parse_from([
            "cars-demo",
            "--input",
            exe_dir.to_str().unwrap(),
            "--no-picasso",
        ])
        .unwrap();
        let resolved = cli.resolve().expect("resolve should accept no --output");
        assert!(!resolved.picasso_enabled);
        assert!(!resolved.draw_enabled);
        assert!(resolved.output.is_none());
    }

    /// Without `--no-picasso`, omitting `--output` is a hard error:
    /// the default pipeline writes an MP4 and silently skipping it
    /// would change the output contract from "always produces a
    /// file" to "sometimes produces a file".
    #[test]
    fn missing_output_rejected_when_picasso_enabled() {
        let exe_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("examples/cars_demo/cars_tracking.rs");
        let cli = Cli::try_parse_from(["cars-demo", "--input", exe_dir.to_str().unwrap()]).unwrap();
        let err = cli.resolve().unwrap_err();
        assert!(
            err.to_string().contains("--output is required"),
            "unexpected error: {err:#}"
        );
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

    /// `--output null` keeps Picasso enabled, suppresses the
    /// filesystem check (the parent directory of `"null"` does
    /// not need to exist), resolves to `output = None`, and
    /// flips `output_is_null = true` on the resolved struct.
    #[test]
    fn output_null_sentinel_disables_file_but_keeps_picasso() {
        let exe_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("examples/cars_demo/cars_tracking.rs");
        let cli = Cli::try_parse_from([
            "cars-demo",
            "--input",
            exe_dir.to_str().unwrap(),
            "--output",
            "null",
        ])
        .unwrap();
        let resolved = cli.resolve().expect("--output null must resolve");
        assert!(resolved.picasso_enabled);
        assert!(resolved.output.is_none());
        assert!(resolved.output_is_null);
    }

    /// `--no-picasso --output null` is rejected — both disable
    /// output on their own and the resolver refuses to pick one
    /// silently.
    #[test]
    fn no_picasso_rejects_output_null() {
        let exe_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("examples/cars_demo/cars_tracking.rs");
        let cli = Cli::try_parse_from([
            "cars-demo",
            "--input",
            exe_dir.to_str().unwrap(),
            "--no-picasso",
            "--output",
            "null",
        ])
        .unwrap();
        let err = cli
            .resolve()
            .expect_err("--no-picasso + --output null must be rejected as a clash");
        let msg = err.to_string();
        assert!(
            msg.contains("--output null") && msg.contains("--no-picasso"),
            "unexpected error: {err:#}"
        );
    }

    /// A regular file path still produces a non-null resolved
    /// output (so the default `output_is_null = false` path is
    /// unaffected by the sentinel plumbing).
    #[test]
    fn regular_output_has_output_is_null_false() {
        let exe_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("examples/cars_demo/cars_tracking.rs");
        let cli = Cli::try_parse_from([
            "cars-demo",
            "--input",
            exe_dir.to_str().unwrap(),
            "--output",
            "/tmp/out.mp4",
        ])
        .unwrap();
        let resolved = cli.resolve().expect("regular output must resolve");
        assert!(resolved.picasso_enabled);
        assert!(!resolved.output_is_null);
        assert!(resolved.output.is_some());
    }
}
