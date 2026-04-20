pub mod generated {
    include!(concat!(env!("OUT_DIR"), "/protocol.rs"));
}

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
