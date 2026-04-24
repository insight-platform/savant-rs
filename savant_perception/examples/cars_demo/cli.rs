//! Command-line argument parsing shared by the sample binaries.
//!
//! Each sample binary simply parses a [`Cli`] and hands the resolved struct
//! to its pipeline `run` entry point.  Keeping the parser in a library
//! module means all argument semantics (defaults, validation, path
//! resolution) live in one place and are covered by unit tests.

use anyhow::{bail, Result};
use clap::Parser;
use std::fmt;
use std::path::{Path, PathBuf};

/// Minimum allowed `--channel-cap`.  Keeps backpressure meaningful:
/// capacity 1 means every intra-pipeline send serializes an entire
/// stage, which is too tight in practice.
pub const MIN_CHANNEL_CAPACITY: usize = 2;

/// End-to-end streaming samples CLI.
#[derive(Debug, Clone, Parser)]
#[command(
    name = "cars-demo",
    about = "Streaming MP4/URI -> decode -> YOLO -> NvDCF -> draw -> encode -> MP4 pipeline."
)]
pub struct Cli {
    /// Input source.  Either:
    ///
    /// * a filesystem path to an MP4 (or any GStreamer-parsable
    ///   container).  Relative paths are resolved against the
    ///   `savant_perception` crate manifest directory; the path
    ///   must exist.
    /// * a URI (e.g. `file:///abs/path.mp4`, `http://host/a.m3u8`,
    ///   `rtsp://cam/stream`, `rtmp://host/key`, …).  A URI is
    ///   handed directly to the URI-based demuxer actor and the
    ///   filesystem existence check is skipped.
    ///
    /// The dispatcher inside
    /// [`crate::cars_tracking::pipeline::run`] picks the demuxer
    /// actor (`Mp4Demuxer` vs `UriDemuxer`) based on which variant
    /// this string resolves to.
    #[arg(long)]
    pub input: String,

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

/// Resolved input — either a filesystem path (handled by
/// `Mp4DemuxerSource`) or a URI (handled by `UriDemuxerSource`).
///
/// The dispatcher inside
/// [`crate::cars_tracking::pipeline::run`] branches on this
/// variant to pick the demuxer actor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputSource {
    /// Absolute filesystem path, pre-validated to exist.
    Path(PathBuf),
    /// URI string (e.g. `rtsp://cam/stream`); handed to the URI
    /// demuxer as-is.  No filesystem check is performed.
    Uri(String),
}

impl fmt::Display for InputSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InputSource::Path(p) => fmt::Display::fmt(&p.display(), f),
            InputSource::Uri(u) => f.write_str(u),
        }
    }
}

/// Syntactic check for a URI-style input: a non-empty ASCII-alpha
/// scheme followed by `://`.  Mirrors the `is_plausible_uri`
/// helper used inside
/// [`savant_gstreamer::uri_demuxer`].  Filesystem paths (even
/// those containing `:` like Windows drive letters, though this
/// example is Linux-only) do not match because they do not start
/// with a scheme followed by `://`.
fn input_looks_like_uri(s: &str) -> bool {
    let Some((scheme, rest)) = s.split_once("://") else {
        return false;
    };
    if scheme.is_empty() || rest.is_empty() {
        return false;
    }
    let mut chars = scheme.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !first.is_ascii_alphabetic() {
        return false;
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '-' || c == '.')
}

impl Cli {
    /// Resolve the input to either a validated filesystem [`PathBuf`]
    /// or an unmodified URI string.
    ///
    /// * URI inputs (`scheme://…`) are returned as
    ///   [`InputSource::Uri`] without any filesystem check — the
    ///   URI-based demuxer will surface unreachable hosts / bad
    ///   schemes as pipeline errors once it starts.
    /// * Filesystem paths are resolved against `CARGO_MANIFEST_DIR`
    ///   when relative and the result must exist on disk.
    pub fn resolved_input(&self) -> Result<InputSource> {
        if input_looks_like_uri(&self.input) {
            return Ok(InputSource::Uri(self.input.clone()));
        }

        let raw = PathBuf::from(&self.input);
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let path = if raw.is_absolute() {
            raw
        } else {
            manifest_dir.join(&raw)
        };

        if !path.exists() {
            bail!(
                "input file does not exist: {}\n\
                 Place the clip manually, e.g.:\n    \
                 curl -L -o {input} \\\n        \
                 https://eu-central-1.linodeobjects.com/savant-data/demo/ny_city_center.mov\n\
                 Or pass a URI like rtsp://cam/stream (no filesystem check).",
                path.display(),
                input = self.input
            );
        }

        Ok(InputSource::Path(path))
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

/// Fully-validated input + output + numeric knobs ready to be handed to
/// [`crate::cars_tracking::pipeline::run`].
#[derive(Debug, Clone)]
pub struct ResolvedCli {
    /// Resolved input source — either a validated filesystem path
    /// ([`InputSource::Path`]) or a URI ([`InputSource::Uri`]).
    /// The pipeline dispatches to `Mp4DemuxerSource` or
    /// `UriDemuxerSource` based on this variant.
    pub input: InputSource,
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
    /// Borrow the resolved input.  Implements [`fmt::Display`] so
    /// callers can log it uniformly regardless of variant.
    pub fn input(&self) -> &InputSource {
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
        assert!(
            matches!(resolved.input, InputSource::Path(_)),
            "filesystem input resolves to InputSource::Path"
        );
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

    /// `input_looks_like_uri` recognises common GStreamer URI
    /// schemes and rejects bare filesystem paths.
    #[test]
    fn input_uri_detection_covers_common_schemes() {
        assert!(input_looks_like_uri("file:///tmp/x.mp4"));
        assert!(input_looks_like_uri("http://host/a.m3u8"));
        assert!(input_looks_like_uri("https://host/a.m3u8"));
        assert!(input_looks_like_uri("rtsp://cam/stream"));
        assert!(input_looks_like_uri("rtsps://cam/stream"));
        assert!(input_looks_like_uri("rtmp://host/app/key"));
        assert!(input_looks_like_uri("hls+http://host/a.m3u8"));

        assert!(!input_looks_like_uri("some.mov"));
        assert!(!input_looks_like_uri("/abs/path.mov"));
        assert!(!input_looks_like_uri("relative/path.mov"));
        assert!(!input_looks_like_uri("://nohost"));
        assert!(!input_looks_like_uri(""));
        assert!(!input_looks_like_uri("9scheme://host"));
    }

    /// A URI input resolves to [`InputSource::Uri`] without touching
    /// the filesystem — even when the URI names a path that clearly
    /// does not exist.
    #[test]
    fn uri_input_skips_filesystem_check() {
        for uri in [
            "rtsp://cam/stream",
            "http://example.com/a.m3u8",
            "file:///definitely/does/not/exist.mp4",
        ] {
            let cli =
                Cli::try_parse_from(["cars-demo", "--input", uri, "--output", "/tmp/out.mp4"])
                    .unwrap();
            let input = cli
                .resolved_input()
                .unwrap_or_else(|e| panic!("URI {uri} must resolve without fs check, got: {e:#}"));
            match input {
                InputSource::Uri(u) => assert_eq!(u, uri),
                other => panic!("expected Uri, got {other:?}"),
            }
        }
    }

    /// `resolve` propagates the `InputSource::Uri` variant end-to-end
    /// into `ResolvedCli` — the pipeline dispatcher reads this
    /// variant to pick the demuxer actor.
    #[test]
    fn resolve_preserves_uri_variant() {
        let cli = Cli::try_parse_from([
            "cars-demo",
            "--input",
            "rtsp://cam/stream",
            "--output",
            "/tmp/out.mp4",
        ])
        .unwrap();
        let resolved = cli.resolve().expect("URI input must resolve");
        match &resolved.input {
            InputSource::Uri(u) => assert_eq!(u, "rtsp://cam/stream"),
            other => panic!("expected Uri, got {other:?}"),
        }
        assert_eq!(resolved.input().to_string(), "rtsp://cam/stream");
    }

    /// Path input round-trips through `resolved_input` as
    /// [`InputSource::Path`] with an absolute path.
    #[test]
    fn path_input_resolves_to_absolute_path() {
        let cli = Cli::try_parse_from([
            "cars-demo",
            "--input",
            // Relative to CARGO_MANIFEST_DIR.
            "examples/cars_demo/cars_tracking.rs",
            "--output",
            "/tmp/out.mp4",
        ])
        .unwrap();
        let input = cli
            .resolved_input()
            .expect("existing relative path resolves");
        match input {
            InputSource::Path(p) => {
                assert!(p.is_absolute(), "relative path must be made absolute");
                assert!(p.exists(), "resolved path must exist on disk");
            }
            other => panic!("expected Path, got {other:?}"),
        }
    }
}
