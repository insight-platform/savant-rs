pub mod capi;
pub mod pipeline;

#[no_mangle]
pub unsafe extern "C" fn enable_log_tracing() {
    let log_env_var_name = "RUST_LOG";
    let log_env_var_level = "trace";
    if std::env::var(log_env_var_name).is_err() {
        std::env::set_var(log_env_var_name, log_env_var_level);
    }
    _ = pretty_env_logger::try_init(); //.expect("Failed to initialize logger");
    log::set_max_level(log::LevelFilter::Trace);
}

#[no_mangle]
pub unsafe fn hello() {
    log::trace!("Hello from savant_capi::hello()");
}
