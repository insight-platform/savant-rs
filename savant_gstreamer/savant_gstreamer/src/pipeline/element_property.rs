use gstreamer as gst;
use gstreamer::prelude::*;

/// Set a GStreamer element property from a string with a panic guard.
///
/// Returns an error string when the property does not exist or setting it fails.
pub fn set_element_property(element: &gst::Element, key: &str, value: &str) -> Result<(), String> {
    if element.find_property(key).is_none() {
        return Err(format!("property '{key}' not found"));
    }
    let elem = element.clone();
    let k = key.to_string();
    let v = value.to_string();
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        elem.set_property_from_str(&k, &v);
    }))
    .map_err(|_| format!("failed to set '{key}' = '{value}'"))?;
    Ok(())
}
