//! Hybrid CLI for the `cars-demo-zmq` example.
//!
//! Three subcommands carve the demo into producer / pipeline /
//! consumer roles:
//!
//! * [`Subcommand::Producer`] — read a local file / URI, push encoded
//!   frames + EOS over ZMQ.
//! * [`Subcommand::Pipeline`] — pull encoded frames over ZMQ, run the
//!   full `cars-demo` middle stack (decode → infer → track → picasso),
//!   push encoded frames back over ZMQ.
//! * [`Subcommand::Consumer`] — pull encoded frames over ZMQ, mux into
//!   a local MP4 file.
//!
//! All subcommands share the same set of numeric knobs (channel
//! capacity, FPS, GPU id, …) where applicable.  Knob defaults match
//! the `cars-demo` CLI so users can swap one binary for the other
//! without re-learning the flags.

use std::path::PathBuf;

use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand as ClapSubcommand};

use crate::cli::MIN_CHANNEL_CAPACITY;

/// Default IPC endpoint shared by the producer (`--zmq-out`) and the
/// pipeline (`--zmq-in`).  Lossless `dealer+connect` ↔ `router+bind`
/// pair: the pipeline binds, the producer connects.
pub const DEFAULT_PRODUCER_ZMQ_OUT: &str = "dealer+connect:ipc:///tmp/savant_demo_in";
/// Default IPC endpoint the pipeline reads from (paired with
/// [`DEFAULT_PRODUCER_ZMQ_OUT`]).
pub const DEFAULT_PIPELINE_ZMQ_IN: &str = "router+bind:ipc:///tmp/savant_demo_in";
/// Default IPC endpoint the pipeline writes to.  Broadcast `pub+bind`
/// — the consumer subscribes on the other side.
pub const DEFAULT_PIPELINE_ZMQ_OUT: &str = "pub+bind:ipc:///tmp/savant_demo_out";
/// Default IPC endpoint the consumer reads from (paired with
/// [`DEFAULT_PIPELINE_ZMQ_OUT`]).
pub const DEFAULT_CONSUMER_ZMQ_IN: &str = "sub+connect:ipc:///tmp/savant_demo_out";

/// Top-level CLI parser for `cars-demo-zmq`.
#[derive(Debug, Clone, Parser)]
#[command(
    name = "cars-demo-zmq",
    about = "Hybrid ZMQ producer / pipeline / consumer demo for cars-demo."
)]
pub struct CliZmq {
    /// Emit per-frame debug logs from this binary's modules.
    #[arg(long, global = true, default_value_t = false)]
    pub debug: bool,

    #[command(subcommand)]
    pub subcommand: Subcommand,
}

/// `cars-demo-zmq` subcommand selector.
///
/// Each variant carries its own argument struct so CLAP can validate
/// the per-role argument set independently.
#[derive(Debug, Clone, ClapSubcommand)]
pub enum Subcommand {
    /// Stream a local file / URI into a ZMQ socket.
    Producer(ProducerArgs),
    /// Run the full cars-demo pipeline between two ZMQ sockets.
    Pipeline(PipelineArgs),
    /// Mux encoded frames received on a ZMQ socket into an MP4.
    Consumer(ConsumerArgs),
}

/// Producer subcommand arguments.
#[derive(Debug, Clone, Parser)]
pub struct ProducerArgs {
    /// Input source — same semantics as `cars-demo --input`: either
    /// a filesystem path or a URI (`file://`, `http://`, `rtsp://`,
    /// `rtmp://`, `hls://`, …).
    #[arg(long)]
    pub input: String,

    /// ZMQ writer URL.  Defaults to [`DEFAULT_PRODUCER_ZMQ_OUT`]
    /// which is wired to connect to the pipeline subcommand's
    /// default `--zmq-in`, so a three-shell IPC run works without
    /// passing this flag at all.
    #[arg(long = "zmq-out", default_value = DEFAULT_PRODUCER_ZMQ_OUT)]
    pub zmq_out: String,

    /// Source id stamped on every emitted frame.  Defaults to
    /// `"cars-demo-zmq"`.
    #[arg(long = "source-id", default_value = "cars-demo-zmq")]
    pub source_id: String,

    /// Inter-stage channel capacity; same semantics as
    /// `cars-demo --channel-cap`.
    #[arg(long = "channel-cap", default_value_t = 4)]
    pub channel_cap: usize,

    /// Suppress the terminating
    /// [`EncodedMsg::SourceEos`](savant_perception::envelopes::EncodedMsg::SourceEos)
    /// marker (and therefore the ZMQ wire EOS) at the end of the
    /// run.  By default the demuxer forwards `SourceEos` once the
    /// input is exhausted, [`ZmqSink`](savant_perception::stages::ZmqSink)
    /// converts that into a wire EOS, and the producer exits.  With
    /// `--no-eos` the demuxer simply stops forwarding frames; the
    /// pipeline / consumer downstream observe the silence but never
    /// see a wire EOS, so they do not transition to a "drained"
    /// state.  The producer itself still exits naturally — the
    /// supervisor broadcasts Shutdown as soon as the demuxer source
    /// completes.
    ///
    /// Useful when feeding a long-running pipeline that must keep
    /// the same `--source-id` across multiple back-to-back producer
    /// runs without inducing per-source teardown each time.
    #[arg(long = "no-eos", default_value_t = false)]
    pub no_eos: bool,

    /// Loop the input forever — when the underlying demuxer
    /// reaches end-of-stream, the producer immediately restarts a
    /// fresh demuxer on the same `--input` and `--source-id`.  The
    /// producer only exits on `Ctrl+C` (or a fatal demuxer error).
    ///
    /// `--loop` is **independent of** `--no-eos`: by default each
    /// loop iteration still forwards its terminating
    /// [`EncodedMsg::SourceEos`](savant_perception::envelopes::EncodedMsg::SourceEos),
    /// which the [`ZmqSink`](savant_perception::stages::ZmqSink)
    /// turns into a wire EOS.  Combine `--loop --no-eos` to suppress
    /// per-iteration EOS markers when feeding a long-lived consumer
    /// that should not transition to a "drained" state between
    /// iterations.
    ///
    /// Useful for sustained load tests against a long-running
    /// pipeline / consumer pair, or for replaying a short clip
    /// against a pipeline that cannot be re-started easily.
    #[arg(long = "loop", default_value_t = false)]
    pub loop_input: bool,
}

/// Consumer subcommand arguments.
#[derive(Debug, Clone, Parser)]
pub struct ConsumerArgs {
    /// ZMQ reader URL.  Defaults to [`DEFAULT_CONSUMER_ZMQ_IN`]
    /// which subscribes to the pipeline subcommand's default
    /// `--zmq-out`, so a three-shell IPC run works without passing
    /// this flag at all.
    #[arg(long = "zmq-in", default_value = DEFAULT_CONSUMER_ZMQ_IN)]
    pub zmq_in: String,

    /// Output `.mp4` file.  Parent directory must exist.
    #[arg(long)]
    pub output: PathBuf,

    /// Output framerate numerator.  Same semantics as
    /// `cars-demo --fps-num`.
    #[arg(long = "fps-num", default_value_t = 25)]
    pub fps_num: i32,

    /// Output framerate denominator.  Must be >= 1.
    #[arg(long = "fps-den", default_value_t = 1)]
    pub fps_den: i32,

    /// Inter-stage channel capacity.
    #[arg(long = "channel-cap", default_value_t = 4)]
    pub channel_cap: usize,
}

/// Pipeline subcommand arguments.
#[derive(Debug, Clone, Parser)]
pub struct PipelineArgs {
    /// ZMQ reader URL (encoded frames in).  Defaults to
    /// [`DEFAULT_PIPELINE_ZMQ_IN`] — the pipeline binds and the
    /// producer's default `--zmq-out` connects to it.
    #[arg(long = "zmq-in", default_value = DEFAULT_PIPELINE_ZMQ_IN)]
    pub zmq_in: String,

    /// ZMQ writer URL (encoded frames out).  Defaults to
    /// [`DEFAULT_PIPELINE_ZMQ_OUT`] — the pipeline binds and the
    /// consumer's default `--zmq-in` connects to it.
    #[arg(long = "zmq-out", default_value = DEFAULT_PIPELINE_ZMQ_OUT)]
    pub zmq_out: String,

    /// CUDA device ID.
    #[arg(long, default_value_t = 0)]
    pub gpu: u32,

    /// Detection confidence threshold.
    #[arg(long, default_value_t = 0.25)]
    pub conf: f32,

    /// NMS IoU threshold.
    #[arg(long, default_value_t = 0.45)]
    pub iou: f32,

    /// Inter-stage channel capacity.
    #[arg(long = "channel-cap", default_value_t = 4)]
    pub channel_cap: usize,

    /// Output framerate numerator (used by the encoder rate-control
    /// inside Picasso).
    #[arg(long = "fps-num", default_value_t = 25)]
    pub fps_num: i32,

    /// Output framerate denominator.
    #[arg(long = "fps-den", default_value_t = 1)]
    pub fps_den: i32,

    /// Disable Picasso's draw stage (overlay).  Picasso still runs
    /// the GPU transform + encoder; only the Skia overlay is
    /// skipped.
    #[arg(long = "no-draw", default_value_t = false)]
    pub no_draw: bool,

    /// Drop Picasso *and* the trailing
    /// [`ZmqSink`](savant_perception::stages::ZmqSink) so the
    /// pipeline terminates at the tracker output: the
    /// [`Function`](savant_perception::stages::Function) terminus
    /// counts inference / tracker frames after the tracker stage,
    /// and no GPU transform / encoder / Skia overlay run at all.
    /// This is stricter than `cars-demo --output null` (which keeps
    /// Picasso alive and only drops the bitstream); use it to
    /// measure raw decode → infer → track throughput, free of any
    /// encode / overlay cost.
    ///
    /// Implies [`Self::no_draw`] because the Skia overlay is part of
    /// the Picasso stage that no longer exists in this mode.
    /// Combined with `producer --no-eos` it provides a sustained
    /// back-to-back load test where the matching consumer would
    /// otherwise add backpressure.
    ///
    /// Multi-stream semantics are preserved: per-source `SourceEos`
    /// is logged but does not terminate the terminus, so the
    /// pipeline keeps running across multiple producer cycles and
    /// only stops on `Ctrl+C`.
    #[arg(long = "no-sink", default_value_t = false)]
    pub no_sink: bool,

    /// Period (seconds) between time-based stats reports while the
    /// pipeline runs.  A final report is always emitted on shutdown
    /// regardless of the period.  Default:
    /// [`DEFAULT_STATS_PERIOD_SECS`](crate::cars_demo::pipeline::DEFAULT_STATS_PERIOD_SECS).
    /// Must be `>= 1`.
    #[arg(long = "stats-period", default_value_t = crate::cars_demo::pipeline::DEFAULT_STATS_PERIOD_SECS)]
    pub stats_period: u64,
}

impl ProducerArgs {
    /// Validate numeric knobs.
    pub fn validate(&self) -> Result<()> {
        if self.channel_cap < MIN_CHANNEL_CAPACITY {
            return Err(anyhow!("--channel-cap must be >= {MIN_CHANNEL_CAPACITY}"));
        }
        if self.input.trim().is_empty() {
            return Err(anyhow!("--input must not be empty"));
        }
        if self.zmq_out.trim().is_empty() {
            return Err(anyhow!("--zmq-out must not be empty"));
        }
        if self.source_id.trim().is_empty() {
            return Err(anyhow!("--source-id must not be empty"));
        }
        Ok(())
    }
}

impl ConsumerArgs {
    /// Validate numeric knobs and output path parent.
    pub fn validate(&self) -> Result<()> {
        if self.channel_cap < MIN_CHANNEL_CAPACITY {
            return Err(anyhow!("--channel-cap must be >= {MIN_CHANNEL_CAPACITY}"));
        }
        if self.fps_num <= 0 {
            return Err(anyhow!("--fps-num must be > 0, got {}", self.fps_num));
        }
        if self.fps_den <= 0 {
            return Err(anyhow!("--fps-den must be > 0, got {}", self.fps_den));
        }
        if self.zmq_in.trim().is_empty() {
            return Err(anyhow!("--zmq-in must not be empty"));
        }
        if let Some(parent) = self.output.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                return Err(anyhow!(
                    "output directory does not exist: {}",
                    parent.display()
                ));
            }
        }
        Ok(())
    }
}

impl PipelineArgs {
    /// Validate numeric knobs.
    pub fn validate(&self) -> Result<()> {
        if self.channel_cap < MIN_CHANNEL_CAPACITY {
            return Err(anyhow!("--channel-cap must be >= {MIN_CHANNEL_CAPACITY}"));
        }
        if !(0.0..=1.0).contains(&self.conf) {
            return Err(anyhow!("--conf must be in [0, 1], got {}", self.conf));
        }
        if !(0.0..=1.0).contains(&self.iou) {
            return Err(anyhow!("--iou must be in [0, 1], got {}", self.iou));
        }
        if self.fps_num <= 0 {
            return Err(anyhow!("--fps-num must be > 0, got {}", self.fps_num));
        }
        if self.fps_den <= 0 {
            return Err(anyhow!("--fps-den must be > 0, got {}", self.fps_den));
        }
        if self.zmq_in.trim().is_empty() {
            return Err(anyhow!("--zmq-in must not be empty"));
        }
        if self.zmq_out.trim().is_empty() {
            return Err(anyhow!("--zmq-out must not be empty"));
        }
        if self.stats_period == 0 {
            return Err(anyhow!(
                "--stats-period must be >= 1 second, got {}",
                self.stats_period
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn parse_producer_subcommand() {
        let cli = CliZmq::try_parse_from([
            "cars-demo-zmq",
            "producer",
            "--input",
            "rtsp://cam/stream",
            "--zmq-out",
            "pub+bind:ipc:///tmp/savant_demo_in",
        ])
        .unwrap();
        match cli.subcommand {
            Subcommand::Producer(args) => {
                assert_eq!(args.input, "rtsp://cam/stream");
                assert_eq!(args.zmq_out, "pub+bind:ipc:///tmp/savant_demo_in");
                assert_eq!(args.source_id, "cars-demo-zmq");
                args.validate().unwrap();
            }
            other => panic!("expected Producer, got {other:?}"),
        }
    }

    #[test]
    fn parse_consumer_subcommand() {
        let cli = CliZmq::try_parse_from([
            "cars-demo-zmq",
            "consumer",
            "--zmq-in",
            "sub+connect:ipc:///tmp/savant_demo_out",
            "--output",
            "/tmp/cars_zmq.mp4",
        ])
        .unwrap();
        match cli.subcommand {
            Subcommand::Consumer(args) => {
                assert_eq!(args.zmq_in, "sub+connect:ipc:///tmp/savant_demo_out");
                assert_eq!(args.output, PathBuf::from("/tmp/cars_zmq.mp4"));
                assert_eq!(args.fps_num, 25);
                args.validate().unwrap();
            }
            other => panic!("expected Consumer, got {other:?}"),
        }
    }

    #[test]
    fn parse_pipeline_subcommand() {
        let cli = CliZmq::try_parse_from([
            "cars-demo-zmq",
            "pipeline",
            "--zmq-in",
            "sub+connect:ipc:///tmp/savant_demo_in",
            "--zmq-out",
            "pub+bind:ipc:///tmp/savant_demo_out",
        ])
        .unwrap();
        match cli.subcommand {
            Subcommand::Pipeline(args) => {
                assert_eq!(args.zmq_in, "sub+connect:ipc:///tmp/savant_demo_in");
                assert_eq!(args.zmq_out, "pub+bind:ipc:///tmp/savant_demo_out");
                assert_eq!(args.gpu, 0);
                assert!((args.conf - 0.25).abs() < 1e-6);
                args.validate().unwrap();
            }
            other => panic!("expected Pipeline, got {other:?}"),
        }
    }

    #[test]
    fn producer_zmq_out_defaults_when_omitted() {
        let cli = CliZmq::try_parse_from(["cars-demo-zmq", "producer", "--input", "/tmp/clip.mp4"])
            .unwrap();
        match cli.subcommand {
            Subcommand::Producer(args) => {
                assert_eq!(args.zmq_out, DEFAULT_PRODUCER_ZMQ_OUT);
                args.validate().unwrap();
            }
            other => panic!("expected Producer, got {other:?}"),
        }
    }

    #[test]
    fn consumer_zmq_in_defaults_when_omitted() {
        let cli =
            CliZmq::try_parse_from(["cars-demo-zmq", "consumer", "--output", "/tmp/cars_zmq.mp4"])
                .unwrap();
        match cli.subcommand {
            Subcommand::Consumer(args) => {
                assert_eq!(args.zmq_in, DEFAULT_CONSUMER_ZMQ_IN);
                args.validate().unwrap();
            }
            other => panic!("expected Consumer, got {other:?}"),
        }
    }

    #[test]
    fn pipeline_endpoints_default_when_omitted() {
        let cli = CliZmq::try_parse_from(["cars-demo-zmq", "pipeline"]).unwrap();
        match cli.subcommand {
            Subcommand::Pipeline(args) => {
                assert_eq!(args.zmq_in, DEFAULT_PIPELINE_ZMQ_IN);
                assert_eq!(args.zmq_out, DEFAULT_PIPELINE_ZMQ_OUT);
                args.validate().unwrap();
            }
            other => panic!("expected Pipeline, got {other:?}"),
        }
    }

    #[test]
    fn defaults_form_a_compatible_chain() {
        // Producer connects to the same endpoint the pipeline binds on, and the
        // consumer connects to the same endpoint the pipeline broadcasts on.
        // Strip the socket-type prefix and compare the underlying endpoint.
        fn endpoint(url: &str) -> &str {
            url.split_once(':').map(|(_opts, rest)| rest).unwrap_or(url)
        }
        assert_eq!(
            endpoint(DEFAULT_PRODUCER_ZMQ_OUT),
            endpoint(DEFAULT_PIPELINE_ZMQ_IN),
        );
        assert_eq!(
            endpoint(DEFAULT_PIPELINE_ZMQ_OUT),
            endpoint(DEFAULT_CONSUMER_ZMQ_IN),
        );
    }

    #[test]
    fn rejects_small_channel_cap() {
        let args = ProducerArgs {
            input: "x".into(),
            zmq_out: "y".into(),
            source_id: "s".into(),
            channel_cap: 0,
            no_eos: false,
            loop_input: false,
        };
        assert!(args.validate().is_err());
    }

    #[test]
    fn producer_no_eos_defaults_off() {
        let cli = CliZmq::try_parse_from(["cars-demo-zmq", "producer", "--input", "/tmp/clip.mp4"])
            .unwrap();
        match cli.subcommand {
            Subcommand::Producer(args) => {
                assert!(!args.no_eos);
                args.validate().unwrap();
            }
            other => panic!("expected Producer, got {other:?}"),
        }
    }

    #[test]
    fn producer_no_eos_can_be_enabled() {
        let cli = CliZmq::try_parse_from([
            "cars-demo-zmq",
            "producer",
            "--input",
            "/tmp/clip.mp4",
            "--no-eos",
        ])
        .unwrap();
        match cli.subcommand {
            Subcommand::Producer(args) => {
                assert!(args.no_eos);
                args.validate().unwrap();
            }
            other => panic!("expected Producer, got {other:?}"),
        }
    }

    #[test]
    fn producer_loop_defaults_off() {
        let cli = CliZmq::try_parse_from(["cars-demo-zmq", "producer", "--input", "/tmp/clip.mp4"])
            .unwrap();
        match cli.subcommand {
            Subcommand::Producer(args) => {
                assert!(!args.loop_input);
                args.validate().unwrap();
            }
            other => panic!("expected Producer, got {other:?}"),
        }
    }

    #[test]
    fn producer_loop_can_be_enabled() {
        let cli = CliZmq::try_parse_from([
            "cars-demo-zmq",
            "producer",
            "--input",
            "/tmp/clip.mp4",
            "--loop",
        ])
        .unwrap();
        match cli.subcommand {
            Subcommand::Producer(args) => {
                assert!(args.loop_input);
                args.validate().unwrap();
            }
            other => panic!("expected Producer, got {other:?}"),
        }
    }

    /// `--no-sink` defaults to `false` so the legacy three-shell
    /// IPC chain (producer → pipeline → consumer) keeps working
    /// without any extra flags.
    #[test]
    fn pipeline_no_sink_defaults_off() {
        let cli = CliZmq::try_parse_from(["cars-demo-zmq", "pipeline"]).unwrap();
        match cli.subcommand {
            Subcommand::Pipeline(args) => {
                assert!(!args.no_sink);
                args.validate().unwrap();
            }
            other => panic!("expected Pipeline, got {other:?}"),
        }
    }

    /// `--no-sink` flips the bool; the rest of the pipeline knobs
    /// keep their defaults so the no-sink benchmark mode is
    /// reachable with a single extra flag.
    #[test]
    fn pipeline_no_sink_can_be_enabled() {
        let cli = CliZmq::try_parse_from(["cars-demo-zmq", "pipeline", "--no-sink"]).unwrap();
        match cli.subcommand {
            Subcommand::Pipeline(args) => {
                assert!(args.no_sink);
                assert_eq!(args.zmq_in, DEFAULT_PIPELINE_ZMQ_IN);
                args.validate().unwrap();
            }
            other => panic!("expected Pipeline, got {other:?}"),
        }
    }
}
