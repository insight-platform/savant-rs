pub mod default_once;
pub mod iter;
use std::fmt::Write;

pub fn bytes_to_hex_string(bytes: &[u8]) -> String {
    bytes.iter().fold(String::new(), |mut output, b| {
        let _ = write!(output, "{b:02X}");
        output
    })
}
