use std::{env, path::PathBuf};

fn glib_arch_include_dir() -> String {
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let multiarch = match arch.as_str() {
        "x86_64" => "x86_64-linux-gnu",
        "aarch64" => "aarch64-linux-gnu",
        other => panic!("Unsupported target architecture: {}", other),
    };
    format!("/usr/lib/{}/glib-2.0/include", multiarch)
}

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let gstnvdsmeta_out_path = out_dir.join("gstnvdsmeta.rs");

    let glib_arch_include = glib_arch_include_dir();

    // Use the version-agnostic symlink so this works across DS 7.0 (x86) and 7.1 (Jetson).
    let ds_dir = "/opt/nvidia/deepstream/deepstream";

    println!("cargo:include={}/sources/includes", ds_dir);
    println!("cargo:include=/usr/include/glib-2.0");
    println!("cargo:include={}", glib_arch_include);
    println!("cargo:include=/usr/include/gstreamer-1.0");
    println!("cargo:rustc-link-search=native={}/lib/", ds_dir);
    println!("cargo:rustc-link-lib=nvdsgst_meta");
    println!("cargo:rustc-link-lib=nvds_meta");
    println!("cargo:rustc-link-lib=nvds_infer");

    println!("cargo:rerun-if-changed=gstnvdsmeta_rs.h");
    let bindings = bindgen::Builder::default()
        .header("gstnvdsmeta_rs.h")
        .clang_arg(format!("-I{}/sources/includes", ds_dir))
        .clang_arg("-I/usr/include/glib-2.0")
        .clang_arg(format!("-I{}", glib_arch_include))
        .clang_arg("-I/usr/include/gstreamer-1.0")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    // Write bindings into OUT_DIR; lib.rs pulls them in via `include!`.
    bindings
        .write_to_file(&gstnvdsmeta_out_path)
        .expect("Couldn't write bindings!");
}
