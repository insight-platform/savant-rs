fn main() {
    // When the `python` feature is enabled, emit PyO3 extension-module link args.
    if std::env::var("CARGO_FEATURE_PYTHON").is_ok() {
        pyo3_build_config::add_extension_module_link_args();
    }
}
