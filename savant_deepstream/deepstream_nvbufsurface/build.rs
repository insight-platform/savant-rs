use std::{env, path::PathBuf};

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ds_dir = "/opt/nvidia/deepstream/deepstream";

    // Link against DeepStream buffer pool library
    println!("cargo:rustc-link-search=native={}/lib/", ds_dir);
    println!("cargo:rustc-link-lib=nvdsbufferpool");

    // Link against CUDA runtime (needed for cuda_init and transitively by nvdsbufferpool)
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");

    // When the `skia` feature is enabled, also link against EGL and GL
    // for CUDA-GL interop (headless rendering via EGL + Skia GL backend).
    if env::var("CARGO_FEATURE_SKIA").is_ok() {
        println!("cargo:rustc-link-lib=EGL");
        println!("cargo:rustc-link-lib=GL");
    }

    // When the `python` feature is enabled, emit PyO3 extension-module link args.
    if env::var("CARGO_FEATURE_PYTHON").is_ok() {
        pyo3_build_config::add_extension_module_link_args();
    }

    // Rebuild if header changes
    println!("cargo:rerun-if-changed=nvbufsurface_rs.h");

    // Generate bindings for NvBufSurface types only.
    // We do NOT include gstnvdsbufferpool.h here to avoid pulling in GStreamer
    // types that would conflict with gstreamer-rs. The gst_nvds_buffer_pool_new()
    // function is manually declared in src/ffi.rs using gstreamer-rs types.
    let bindings = bindgen::Builder::default()
        .header("nvbufsurface_rs.h")
        .clang_arg(format!("-I{}/sources/includes", ds_dir))
        .allowlist_type("NvBufSurface")
        .allowlist_type("NvBufSurfaceMemType")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate NvBufSurface bindings");

    bindings
        .write_to_file(out_dir.join("nvbufsurface_bindings.rs"))
        .expect("Couldn't write NvBufSurface bindings!");
}
