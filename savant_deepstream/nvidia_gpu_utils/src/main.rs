//! CLI utility to query GPU/device info for use in shell scripts.
//!
//! Outputs `KEY='value'` pairs (shell-safe) for easy sourcing:
//!
//! ```bash
//! # Source into current shell
//! eval $(nvidia_gpu_info jetson_model gpu_mem_used_mib)
//! echo "Model: $NVIDIA_GPU_JETSON_MODEL"
//!
//! # Or query specific params with custom GPU
//! eval $(nvidia_gpu_info -g 0 is_jetson is_orin_nano mem_total_mib)
//! if [ "$NVIDIA_GPU_IS_ORIN_NANO" = "true" ]; then
//!   echo "Running on Orin Nano, total mem: $NVIDIA_GPU_MEM_TOTAL_MIB MiB"
//! fi
//! ```

use clap::Parser;
use nvidia_gpu_utils::{
    gpu_mem_used_mib, is_jetson_kernel, jetson_model, mem_total_mib, process_rss_mib,
};

/// Query GPU and device info. Outputs shell-safe KEY='value' pairs.
///
/// Use with `eval $(nvidia_gpu_info ...)` to set variables in your script.
#[derive(Parser, Debug)]
#[command(name = "nvidia_gpu_info")]
#[command(about = "Query GPU/device info for shell scripts")]
struct Args {
    /// GPU device ID (default: 0)
    #[arg(short, long, default_value = "0")]
    gpu_id: u32,

    /// Parameters to output. If none given, prints all.
    ///
    /// Available: jetson_model, is_jetson, is_jetson_kernel, is_orin_nano,
    /// gpu_mem_used_mib, mem_total_mib, process_rss_mib
    #[arg(value_name = "PARAM")]
    params: Vec<String>,
}

const ALL_PARAMS: &[&str] = &[
    "jetson_model",
    "is_jetson",
    "is_jetson_kernel",
    "is_orin_nano",
    "gpu_mem_used_mib",
    "mem_total_mib",
    "process_rss_mib",
];

fn main() {
    let args = Args::parse();
    let params: Vec<&str> = if args.params.is_empty() {
        ALL_PARAMS.to_vec()
    } else {
        args.params.iter().map(|s| s.as_str()).collect()
    };

    for param in params {
        if let Err(e) = print_param(param, args.gpu_id) {
            eprintln!("nvidia_gpu_info: {}: {}", param, e);
            std::process::exit(1);
        }
    }
}

fn print_param(param: &str, gpu_id: u32) -> Result<(), Box<dyn std::error::Error>> {
    match param {
        "jetson_model" => {
            let val = match jetson_model(gpu_id)? {
                Some(m) => m.to_string(),
                None => String::new(),
            };
            println!("NVIDIA_GPU_JETSON_MODEL='{}'", escape_for_shell(&val));
        }
        "is_jetson" => {
            let val = jetson_model(gpu_id)?.is_some();
            println!("NVIDIA_GPU_IS_JETSON='{}'", val);
        }
        "is_jetson_kernel" => {
            let val = is_jetson_kernel();
            println!("NVIDIA_GPU_IS_JETSON_KERNEL='{}'", val);
        }
        "is_orin_nano" => {
            let val = jetson_model(gpu_id)
                .ok()
                .and_then(|m| m)
                .is_some_and(|m| m.is_orin_nano());
            println!("NVIDIA_GPU_IS_ORIN_NANO='{}'", val);
        }
        "gpu_mem_used_mib" => {
            let val = gpu_mem_used_mib(gpu_id)?;
            println!("NVIDIA_GPU_MEM_USED_MIB='{}'", val);
        }
        "mem_total_mib" => {
            let val = mem_total_mib()?;
            println!("NVIDIA_GPU_MEM_TOTAL_MIB='{}'", val);
        }
        "process_rss_mib" => {
            let val = process_rss_mib()?;
            println!("NVIDIA_GPU_PROCESS_RSS_MIB='{}'", val);
        }
        _ => {
            eprintln!("Unknown parameter: {}", param);
            eprintln!("Available: {}", ALL_PARAMS.join(", "));
            std::process::exit(1);
        }
    }
    Ok(())
}

fn escape_for_shell(s: &str) -> String {
    s.replace('\'', "'\\''")
}
