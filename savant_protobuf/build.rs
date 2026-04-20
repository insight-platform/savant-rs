extern crate prost_build;

use std::env;
use std::path::PathBuf;

fn main() {
    let proto_path = PathBuf::from("src/savant_rs.proto");

    println!("cargo:rerun-if-changed=src/savant_rs.proto");

    // Honour OUT_DIR so generated sources live in the standard Cargo build
    // directory and are not written back into the crate's `src/` tree.
    env::var("OUT_DIR").expect("OUT_DIR must be set by Cargo when running build scripts");

    let mut config = prost_build::Config::new();
    config.protoc_arg("--experimental_allow_proto3_optional");
    config.enum_attribute(".", "#[allow(clippy::large_enum_variant)]");
    config
        .compile_protos(
            &[proto_path.to_str().unwrap()],
            &[proto_path.parent().unwrap().to_str().unwrap()],
        )
        .expect("Failed to compile protobuf definitions");
}
