pub mod generated;

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
