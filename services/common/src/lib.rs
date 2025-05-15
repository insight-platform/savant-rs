pub mod job_writer;
pub mod source;
use savant_core::utils::bytes_to_hex_string;
use std::str::from_utf8;
use std::time::{SystemTime, UNIX_EPOCH};

pub fn topic_to_string(topic: &[u8]) -> String {
    from_utf8(topic)
        .map(String::from)
        .unwrap_or(bytes_to_hex_string(topic))
}

pub fn systime_ms() -> u128 {
    let start = SystemTime::now();
    let since_the_epoch = start
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards");
    since_the_epoch.as_millis()
}
