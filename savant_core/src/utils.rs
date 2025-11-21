pub mod default_once;
pub mod iter;
pub mod rtp_pts_mapper;
pub mod uuid_v7;
use std::fmt::Write;

pub fn bytes_to_hex_string(bytes: &[u8]) -> String {
    bytes.iter().fold(String::new(), |mut output, b| {
        let _ = write!(output, "{b:02X}");
        output
    })
}
