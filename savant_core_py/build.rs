use std::env;

fn main() {
    pyo3_build_config::add_extension_module_link_args();

    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let config = cbindgen::Config {
        language: cbindgen::Language::C,
        ..Default::default()
    };
    cbindgen::generate_with_config(crate_dir, config)
        .unwrap()
        .write_to_file("../savant_python/python/savant_rs/include/savant_rs.h");
}
