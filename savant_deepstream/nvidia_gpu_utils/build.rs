//! Link against libcuda for Jetson model detection (cuDeviceGetAttribute).

fn main() {
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    match arch.as_str() {
        "aarch64" => {
            println!("cargo:rustc-link-search=native=/usr/local/cuda/targets/aarch64-linux/lib");
            println!(
                "cargo:rustc-link-search=native=/usr/local/cuda/targets/aarch64-linux/lib/stubs"
            );
        }
        _ => {}
    }
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    // CUDA stubs for CI builds where the host GPU driver is absent
    // (libcuda.so is normally injected from the host at runtime).
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64/stubs");
    println!("cargo:rustc-link-lib=cuda");
}
