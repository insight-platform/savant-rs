use std::{env, path::PathBuf};

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ds_dir = "/opt/nvidia/deepstream/deepstream";

    // Link against DeepStream libraries
    println!("cargo:rustc-link-search=native={}/lib/", ds_dir);
    println!("cargo:rustc-link-lib=nvdsbufferpool");
    println!("cargo:rustc-link-lib=nvbufsurftransform");

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

    // Rebuild if headers change
    println!("cargo:rerun-if-changed=nvbufsurface_rs.h");
    println!("cargo:rerun-if-changed=nvbufsurftransform_rs.h");

    let clang_ds_include = format!("-I{}/sources/includes", ds_dir);

    // Generate bindings for NvBufSurface types only.
    // We do NOT include gstnvdsbufferpool.h here to avoid pulling in GStreamer
    // types that would conflict with gstreamer-rs. The gst_nvds_buffer_pool_new()
    // function is manually declared in src/ffi.rs using gstreamer-rs types.
    let bindings = bindgen::Builder::default()
        .header("nvbufsurface_rs.h")
        .clang_arg(&clang_ds_include)
        .allowlist_type("NvBufSurface")
        .allowlist_type("NvBufSurfaceMemType")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate NvBufSurface bindings");

    bindings
        .write_to_file(out_dir.join("nvbufsurface_bindings.rs"))
        .expect("Couldn't write NvBufSurface bindings!");

    // Generate bindings for NvBufSurfTransform types and functions.
    let transform_bindings = bindgen::Builder::default()
        .header("nvbufsurftransform_rs.h")
        .clang_arg(&clang_ds_include)
        // Types
        .allowlist_type("NvBufSurfTransformRect")
        .allowlist_type("NvBufSurfTransformParams")
        .allowlist_type("NvBufSurfTransformConfigParams")
        .allowlist_type("NvBufSurfTransform_Compute")
        .allowlist_type("NvBufSurfTransform_Inter")
        .allowlist_type("NvBufSurfTransform_Error")
        .allowlist_type("NvBufSurfTransform_Transform_Flag")
        .allowlist_type("NvBufSurfTransform_Flip")
        // Functions
        .allowlist_function("NvBufSurfTransform")
        .allowlist_function("NvBufSurfTransformSetSessionParams")
        // Pull in dependent NvBufSurface types
        .allowlist_type("NvBufSurface")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate NvBufSurfTransform bindings");

    transform_bindings
        .write_to_file(out_dir.join("nvbufsurftransform_bindings.rs"))
        .expect("Couldn't write NvBufSurfTransform bindings!");
}
