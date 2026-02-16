use std::env;
use std::fs;

fn main() {
    pyo3_build_config::add_extension_module_link_args();

    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let config = cbindgen::Config {
        language: cbindgen::Language::C,
        ..Default::default()
    };
    let bindings = cbindgen::generate_with_config(crate_dir, config).unwrap();

    let mut new_content = Vec::new();
    bindings.write(&mut new_content);

    let header_path = "../savant_python/python/savant_rs/include/savant_rs.h";
    let needs_update = match fs::read(header_path) {
        Ok(existing) => existing != new_content,
        Err(_) => true,
    };
    if needs_update {
        fs::write(header_path, new_content).unwrap();
    }
}
