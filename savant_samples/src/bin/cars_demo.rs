//! `cars-demo` binary: streaming MP4 -> YOLO -> NvDCF -> Picasso -> MP4.
//!
//! Thin wrapper around
//! [`savant_samples::cars_tracking::pipeline::run`] — parses CLI, wires up
//! the logger (optionally raising our own log level to `debug` via
//! `--debug`) and reports the first stage error back to the process.

use anyhow::Result;
use clap::Parser;
use savant_samples::cars_tracking::pipeline;
use savant_samples::cli::Cli;

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Raise the sample's own log level to debug on --debug while keeping
    // third-party crates at info so the terminal doesn't get spammed by
    // GStreamer / TensorRT internals.
    let default_filter = if cli.debug {
        "info,savant_samples=debug,cars_demo=debug"
    } else {
        "info"
    };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(default_filter))
        .init();

    let resolved = cli.resolve()?;
    log::info!(
        "cars-demo: input={} output={} gpu={} conf={} iou={} channel_cap={} fps={}/{} draw_enabled={} debug={}",
        resolved.input().display(),
        resolved.output().display(),
        resolved.gpu,
        resolved.conf,
        resolved.iou,
        resolved.channel_cap,
        resolved.fps_num,
        resolved.fps_den,
        resolved.draw_enabled,
        resolved.debug,
    );

    pipeline::run(resolved)
}
