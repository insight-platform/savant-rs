/// Attribute-based gates for conditional processing.
///
/// When set, the pipeline checks whether the frame carries the specified
/// attribute before proceeding with the corresponding stage.
#[derive(Debug, Clone, Default)]
pub struct ConditionalSpec {
    /// If set, the frame must have `get_attribute(ns, name).is_some()` to be
    /// processed at all; otherwise the frame is dropped.
    pub encode_attribute: Option<(String, String)>,
    /// If set, the frame must have this attribute for the Skia rendering stage
    /// to run; encoding still proceeds without rendering.
    pub render_attribute: Option<(String, String)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_has_no_gates() {
        let c = ConditionalSpec::default();
        assert!(c.encode_attribute.is_none());
        assert!(c.render_attribute.is_none());
    }

    #[test]
    fn with_frame_gate() {
        let c = ConditionalSpec {
            encode_attribute: Some(("inference".into(), "result".into())),
            render_attribute: None,
        };
        let (ns, name) = c.encode_attribute.as_ref().unwrap();
        assert_eq!(ns, "inference");
        assert_eq!(name, "result");
    }
}
