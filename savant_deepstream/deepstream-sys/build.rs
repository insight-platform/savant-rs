use bindgen;
use std::{env, fs, path::PathBuf};

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let src_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("src");
    //let nvdsmeta_out_path = out_dir.join("nvdsmeta.rs");
    let gstnvdsmeta_out_path = out_dir.join("gstnvdsmeta.rs");
    //let nvdsmeta_module_path = src_dir.join("nvdsmeta.rs");
    let gstnvdsmeta_module_path = src_dir.join("gstnvdsmeta.rs");

    println!(
        "cargo:include={}",
        "/opt/nvidia/deepstream/deepstream-7.0/sources/includes"
    );
    println!("cargo:include={}", "/usr/include/glib-2.0");
    println!(
        "cargo:include={}",
        "/usr/lib/x86_64-linux-gnu/glib-2.0/include"
    );
    // /usr/include/gstreamer-1.0
    println!("cargo:include={}", "/usr/include/gstreamer-1.0");
    println!("cargo:rustc-link-search=native=/opt/nvidia/deepstream/deepstream-7.0/lib/");
    println!("cargo:rustc-link-lib=nvdsgst_meta");
    println!("cargo:rustc-link-lib=nvds_infer");
    //println!("cargo:rustc-link-lib=nvds_meta");

    // println!("cargo:rerun-if-changed=nvdsmeta_rs.h");
    // let bindings = bindgen::Builder::default()
    //     .header("nvdsmeta_rs.h")
    //     .clang_arg("-I/opt/nvidia/deepstream/deepstream-7.0/sources/includes")
    //     .clang_arg("-I/usr/include/glib-2.0")
    //     .clang_arg("-I/usr/lib/x86_64-linux-gnu/glib-2.0/include")
    //     .clang_arg("-I/usr/include/gstreamer-1.0")
    //     .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
    //     .generate()
    //     .expect("Unable to generate bindings");

    // bindings
    //     .write_to_file(&nvdsmeta_out_path)
    //     .expect("Couldn't write bindings!");

    //fs::copy(nvdsmeta_out_path, nvdsmeta_module_path).unwrap();

    println!("cargo:rerun-if-changed=gstnvdsmeta_rs.h");
    let bindings = bindgen::Builder::default()
        .header("gstnvdsmeta_rs.h")
        .clang_arg("-I/opt/nvidia/deepstream/deepstream-7.0/sources/includes")
        .clang_arg("-I/usr/include/glib-2.0")
        .clang_arg("-I/usr/lib/x86_64-linux-gnu/glib-2.0/include")
        .clang_arg("-I/usr/include/gstreamer-1.0")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    bindings
        .write_to_file(&gstnvdsmeta_out_path)
        .expect("Couldn't write bindings!");

    fs::copy(gstnvdsmeta_out_path, gstnvdsmeta_module_path).unwrap();
}
