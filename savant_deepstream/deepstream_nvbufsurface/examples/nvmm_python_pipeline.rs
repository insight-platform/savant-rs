//! Runs the NVMM encoding pipeline from a Python subprocess.
//!
//! This example spawns a Python process that uses the `deepstream_nvbufsurface`
//! PyO3 module to allocate NVMM buffers and push them through a GStreamer
//! encoding pipeline, then parses the results.
//!
//! Running the pipeline in a subprocess avoids conflicts between PyGObject's
//! GStreamer bindings and Rust-initialised GStreamer in the same process.
//!
//! Pipeline (inside the Python subprocess):
//! ```text
//! appsrc (memory:NVMM, NV12) → nvv4l2h265enc → h265parse → appsink
//! ```
//!
//! ## SavantIdMeta validation
//!
//! Each frame pushed to `appsrc` carries a random `Frame(id)` via
//! `SavantIdMeta`.  `bridge_savant_id_meta_py` re-attaches the meta
//! after the encoder (which normally drops custom meta).  At `appsink`,
//! the `get_savant_id_meta` function reads the meta from each output
//! buffer and validates that the received IDs match what was sent, using
//! PTS as the correlation key.
//!
//! # Usage
//!
//! ```sh
//! # Default: runs infinitely at 640×480 (Ctrl+C to stop)
//! cargo run --example nvmm_python_pipeline -p deepstream_nvbufsurface
//!
//! # Custom resolution, infinite
//! cargo run --example nvmm_python_pipeline -p deepstream_nvbufsurface -- \
//!     --width 1920 --height 1080
//!
//! # Fixed frame count
//! cargo run --example nvmm_python_pipeline -p deepstream_nvbufsurface -- \
//!     --width 1920 --height 1080 --num-frames 300
//! ```

use clap::Parser;
use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::time::Instant;

/// Run the NVMM encoding pipeline from a Python subprocess.
#[derive(Parser, Debug)]
#[command(name = "nvmm_python_pipeline")]
struct Args {
    /// Frame width in pixels.
    #[arg(long, default_value_t = 640)]
    width: u32,

    /// Frame height in pixels.
    #[arg(long, default_value_t = 480)]
    height: u32,

    /// Framerate numerator (used for PTS calculation only).
    #[arg(long, default_value_t = 30)]
    fps: i32,

    /// Buffer pool size (number of NVMM buffers).
    /// Must be large enough that the encoder does not hold all buffers
    /// simultaneously, which would deadlock the push loop.
    #[arg(long, default_value_t = 8)]
    pool_size: u32,

    /// Number of frames to generate. When omitted the pipeline runs
    /// indefinitely until the Python subprocess is interrupted (Ctrl+C).
    #[arg(long, short)]
    num_frames: Option<u32>,
}

/// Result parsed from the Python subprocess stdout.
#[derive(Debug)]
struct PythonPipelineResult {
    pushed: u64,
    received: u64,
    meta_ok: u64,
    meta_mismatch: u64,
    meta_missing: u64,
}

/// Build and run the Python pipeline subprocess.
///
/// Stderr is inherited (printed to the terminal in real time) so that
/// GStreamer warnings and Python tracebacks are visible immediately.
/// Stdout is captured via a pipe and the last line is parsed for counts.
fn run_pipeline_python(args: &Args) -> PythonPipelineResult {
    let fps_den = 1;
    let duration_ns: u64 = (1_000_000_000u64)
        .checked_div(args.fps as u64)
        .unwrap_or(33_333_333);

    // When num_frames is None the Python script runs an infinite loop
    // with periodic status prints (Ctrl+C / SIGINT to stop).
    let num_frames_py = match args.num_frames {
        Some(n) => n.to_string(),
        None => "None".to_string(),
    };

    let script = format!(
        r#"
import os, sys, signal, threading, time, random

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GstApp
Gst.init(None)

from deepstream_nvbufsurface import (
    NvBufSurfaceGenerator, init_cuda,
    bridge_savant_id_meta_py, get_savant_id_meta,
)

init_cuda(0)

gen = NvBufSurfaceGenerator("NV12", {width}, {height},
                            fps_num={fps_num}, fps_den={fps_den},
                            pool_size={pool_size})

pipeline = Gst.parse_launch(
    'appsrc name=src format=3 stream-type=0 '
    '! nvv4l2h265enc name=enc '
    '! h265parse ! appsink name=sink emit-signals=true sync=false'
)
appsrc = pipeline.get_by_name('src')
appsrc.set_property('caps', Gst.Caps.from_string(gen.nvmm_caps_str()))

# Bridge SavantIdMeta across the encoder (PTS-keyed pad probes)
enc = pipeline.get_by_name('enc')
bridge_savant_id_meta_py(hash(enc))

# ── Bookkeeping ──────────────────────────────────────────────────────
# Map PTS (ns) → expected random ID, populated by the push loop.
pts_to_id = {{}}
pts_lock = threading.Lock()

received = 0
meta_ok = 0
meta_mismatch = 0
meta_missing = 0
stats_lock = threading.Lock()

def on_new_sample(sink):
    global received, meta_ok, meta_mismatch, meta_missing
    sample = sink.emit('pull-sample')
    if not sample:
        return Gst.FlowReturn.OK

    buf = sample.get_buffer()
    pts = buf.pts

    # Read SavantIdMeta from the output buffer
    ids = get_savant_id_meta(hash(buf))

    with pts_lock:
        expected = pts_to_id.pop(pts, None)

    with stats_lock:
        received += 1
        if not ids:
            meta_missing += 1
        else:
            # ids is a list of ("frame"|"batch", int) tuples.
            # We expect exactly one Frame entry matching our random ID.
            got = [v for kind, v in ids if kind == "frame"]
            if expected is not None and len(got) == 1 and got[0] == expected:
                meta_ok += 1
            else:
                meta_mismatch += 1

    return Gst.FlowReturn.OK

appsink = pipeline.get_by_name('sink')
appsink.connect('new-sample', on_new_sample)

ret = pipeline.set_state(Gst.State.PLAYING)
if ret == Gst.StateChangeReturn.FAILURE:
    print('ERROR: pipeline failed to start (NVENC session limit?)', file=sys.stderr)
    sys.exit(1)

appsrc_ptr = hash(appsrc)
pushed = 0
num_frames = {num_frames}
stopping = False

def handle_sigint(sig, frame):
    global stopping
    stopping = True

signal.signal(signal.SIGINT, handle_sigint)

if num_frames is None:
    print('Running indefinitely (Ctrl+C to stop)...', flush=True)
else:
    print(f'Pushing {{num_frames}} frames...', flush=True)

t0 = time.monotonic()
last_report = t0
i = 0

while True:
    if stopping:
        break
    if num_frames is not None and i >= num_frames:
        break

    # Generate a random ID for this frame
    frame_id = random.randint(-2**62, 2**62)
    pts_ns = i * {duration_ns}

    with pts_lock:
        pts_to_id[pts_ns] = frame_id

    try:
        gen.push_to_appsrc(appsrc_ptr,
                           pts_ns=pts_ns,
                           duration_ns={duration_ns},
                           id=frame_id)
        pushed += 1
    except Exception as e:
        print(f'Push failed at frame {{i}}: {{e}}', file=sys.stderr)
        with pts_lock:
            pts_to_id.pop(pts_ns, None)
        break

    i += 1

    # Periodic status (every second)
    now = time.monotonic()
    if now - last_report >= 1.0:
        elapsed = now - t0
        with stats_lock:
            r, ok, mis, miss = received, meta_ok, meta_mismatch, meta_missing
        fps = pushed / elapsed if elapsed > 0 else 0
        try:
            print(
                f'STATUS pushed={{pushed}} received={{r}} meta_ok={{ok}} '
                f'mismatch={{mis}} missing={{miss}} fps={{fps:.1f}}',
                flush=True,
            )
        except BrokenPipeError:
            break
        last_report = now

if not stopping:
    NvBufSurfaceGenerator.send_eos(appsrc_ptr)
    bus = pipeline.get_bus()
    msg = bus.timed_pop_filtered(30 * Gst.SECOND,
                                 Gst.MessageType.EOS | Gst.MessageType.ERROR)
    if msg and msg.type == Gst.MessageType.ERROR:
        err, debug = msg.parse_error()
        print(f'ERROR: {{err}}', file=sys.stderr)
        pipeline.set_state(Gst.State.NULL)
        sys.exit(1)
else:
    print('Stopping...', flush=True)

pipeline.set_state(Gst.State.NULL)
with stats_lock:
    r, ok, mis, miss = received, meta_ok, meta_mismatch, meta_missing
try:
    print(f'RESULT {{pushed}} {{r}} {{ok}} {{mis}} {{miss}}', flush=True)
except BrokenPipeError:
    pass
"#,
        width = args.width,
        height = args.height,
        fps_num = args.fps,
        fps_den = fps_den,
        pool_size = args.pool_size,
        num_frames = num_frames_py,
        duration_ns = duration_ns,
    );

    let mut child = Command::new("python3")
        .arg("-c")
        .arg(&script)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit()) // show errors in real time
        .spawn()
        .expect("Failed to spawn Python subprocess");

    // Read stdout line-by-line so we can show progress and capture the result.
    let stdout = child.stdout.take().expect("stdout piped");
    let reader = BufReader::new(stdout);
    let mut last_line = String::new();

    for line in reader.lines() {
        let line = line.expect("UTF-8 stdout");
        if line.starts_with("STATUS ") {
            // Pretty-print periodic status from the Python side.
            if let Some(rest) = line.strip_prefix("STATUS ") {
                let kvs: Vec<&str> = rest.split_whitespace().collect();
                println!("  {}", kvs.join("  |  "));
            }
        } else {
            println!("  [python] {}", line);
        }
        last_line = line;
    }

    let status = child.wait().expect("wait for child");
    if !status.success() {
        // SIGINT (exit code 2 on Python, or signal) is expected for infinite mode.
        if args.num_frames.is_none() {
            // Graceful shutdown via Ctrl+C — parse whatever we have.
        } else {
            panic!(
                "Python pipeline exited with status {} (see stderr above)",
                status,
            );
        }
    }

    // Parse the last RESULT line: "RESULT <pushed> <received> <meta_ok> <mismatch> <missing>"
    let mut pushed = 0u64;
    let mut received = 0u64;
    let mut meta_ok = 0u64;
    let mut meta_mismatch = 0u64;
    let mut meta_missing = 0u64;
    if let Some(rest) = last_line.strip_prefix("RESULT ") {
        let mut parts = rest.split_whitespace();
        pushed = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
        received = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
        meta_ok = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
        meta_mismatch = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
        meta_missing = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    }

    PythonPipelineResult {
        pushed,
        received,
        meta_ok,
        meta_mismatch,
        meta_missing,
    }
}

fn main() {
    let args = Args::parse();

    let frames_str = match args.num_frames {
        Some(n) => format!("{} frames", n),
        None => "infinite (Ctrl+C to stop)".to_string(),
    };

    println!(
        "Running Python NVMM pipeline: {}x{} @ {} fps, {} (pool={})",
        args.width, args.height, args.fps, frames_str, args.pool_size,
    );
    println!("Pipeline: appsrc(NV12 NVMM) -> nvv4l2h265enc -> h265parse -> appsink\n");

    let start = Instant::now();
    let result = run_pipeline_python(&args);
    let elapsed = start.elapsed();

    println!(
        "\nResult: pushed={}, received={}, meta_ok={}, mismatch={}, missing={}",
        result.pushed, result.received, result.meta_ok, result.meta_mismatch, result.meta_missing,
    );
    println!("Elapsed: {:.2}s", elapsed.as_secs_f64());

    if result.pushed > 0 {
        let fps = result.pushed as f64 / elapsed.as_secs_f64();
        println!("Throughput: {:.1} fps", fps);
    }

    if let Some(expected) = args.num_frames {
        assert_eq!(
            result.pushed, expected as u64,
            "Not all frames were pushed",
        );
        assert!(
            result.received > 0,
            "appsink received 0 buffers (pushed {})",
            result.pushed,
        );
        assert!(
            result.meta_ok > 0,
            "No buffers carried valid SavantIdMeta (received {})",
            result.received,
        );
        assert_eq!(
            result.meta_mismatch, 0,
            "SavantIdMeta ID mismatch detected ({} mismatches)",
            result.meta_mismatch,
        );
        assert_eq!(
            result.meta_missing, 0,
            "SavantIdMeta missing on {} buffers",
            result.meta_missing,
        );
    }

    println!("\nPython pipeline OK.");
}
