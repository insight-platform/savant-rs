//! Pre-parsed label template engine with `{variable}` syntax.
//!
//! Templates are parsed once at spec-set time into a list of [`Segment`]s,
//! then expanded per-object at render time with minimal allocations.
//!
//! Supported variables:
//!
//! | Variable      | Source                                  |
//! |---------------|-----------------------------------------|
//! | `namespace`   | `obj.get_namespace()`                   |
//! | `label`       | `obj.get_label()`                       |
//! | `draw_label`  | `obj.get_draw_label()` or label         |
//! | `id`          | `obj.get_id()`                          |
//! | `parent_id`   | `obj.get_parent_id()` or empty          |
//! | `track_id`    | `obj.get_track_id()` or empty           |
//! | `confidence`  | `obj.get_confidence()` or empty         |
//! | `det_xc`      | detection box center x                  |
//! | `det_yc`      | detection box center y                  |
//! | `det_width`   | detection box width                     |
//! | `det_height`  | detection box height                    |
//! | `det_angle`   | detection box angle or empty            |
//! | `track_xc`    | tracking box center x or empty          |
//! | `track_yc`    | tracking box center y or empty          |
//! | `track_width` | tracking box width or empty             |
//! | `track_height`| tracking box height or empty            |
//! | `track_angle` | tracking box angle or empty             |

use crate::primitives::object::{BorrowedVideoObject, ObjectOperations};
use anyhow::{bail, Result};
use std::collections::hash_map::DefaultHasher;
use std::fmt::Write;
use std::hash::{Hash, Hasher};

/// Compute a stable `u64` hash for a slice of format strings.
///
/// Used by [`ParsedLabelFormats`] to detect whether the source templates
/// have changed without re-parsing.
///
/// ```
/// use savant_core::label_template::format_hash;
///
/// let a = format_hash(&["hello".into(), "world".into()]);
/// let b = format_hash(&["hello".into(), "world".into()]);
/// assert_eq!(a, b);
///
/// let c = format_hash(&["different".into()]);
/// assert_ne!(a, c);
/// ```
pub fn format_hash(format: &[String]) -> u64 {
    let mut h = DefaultHasher::new();
    format.hash(&mut h);
    h.finish()
}

/// A recognized template variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemplateVar {
    Namespace,
    Label,
    DrawLabel,
    Id,
    ParentId,
    TrackId,
    Confidence,
    DetXc,
    DetYc,
    DetWidth,
    DetHeight,
    DetAngle,
    TrackXc,
    TrackYc,
    TrackWidth,
    TrackHeight,
    TrackAngle,
}

impl TemplateVar {
    fn from_name(name: &str) -> Option<Self> {
        match name {
            "namespace" => Some(Self::Namespace),
            "label" => Some(Self::Label),
            "draw_label" => Some(Self::DrawLabel),
            "id" => Some(Self::Id),
            "parent_id" => Some(Self::ParentId),
            "track_id" => Some(Self::TrackId),
            "confidence" => Some(Self::Confidence),
            "det_xc" => Some(Self::DetXc),
            "det_yc" => Some(Self::DetYc),
            "det_width" => Some(Self::DetWidth),
            "det_height" => Some(Self::DetHeight),
            "det_angle" => Some(Self::DetAngle),
            "track_xc" => Some(Self::TrackXc),
            "track_yc" => Some(Self::TrackYc),
            "track_width" => Some(Self::TrackWidth),
            "track_height" => Some(Self::TrackHeight),
            "track_angle" => Some(Self::TrackAngle),
            _ => None,
        }
    }
}

/// A segment of a parsed template: either literal text or a variable reference.
#[derive(Debug, Clone, PartialEq)]
pub enum Segment {
    Literal(String),
    Variable(TemplateVar),
}

/// A single pre-parsed label template line.
#[derive(Debug, Clone)]
pub struct LabelTemplate {
    segments: Vec<Segment>,
}

impl LabelTemplate {
    /// Parse a template string like `"{namespace}/{label} #{id}"`.
    ///
    /// Returns an error if an unknown variable name is encountered or braces
    /// are unmatched.
    ///
    /// ```
    /// use savant_core::label_template::{LabelTemplate, Segment, TemplateVar};
    ///
    /// let t = LabelTemplate::parse("{namespace}/{label}").unwrap();
    /// assert_eq!(
    ///     t.segments(),
    ///     &[
    ///         Segment::Variable(TemplateVar::Namespace),
    ///         Segment::Literal("/".into()),
    ///         Segment::Variable(TemplateVar::Label),
    ///     ]
    /// );
    /// ```
    pub fn parse(s: &str) -> Result<Self> {
        let mut segments = Vec::new();
        let mut literal = String::new();
        let mut chars = s.chars().peekable();

        while let Some(ch) = chars.next() {
            if ch == '{' {
                if chars.peek() == Some(&'{') {
                    literal.push('{');
                    chars.next();
                    continue;
                }
                if !literal.is_empty() {
                    segments.push(Segment::Literal(std::mem::take(&mut literal)));
                }
                let mut var_name = String::new();
                loop {
                    match chars.next() {
                        Some('}') => break,
                        Some(c) => var_name.push(c),
                        None => bail!("unclosed '{{' in template: \"{s}\""),
                    }
                }
                let var = TemplateVar::from_name(var_name.trim()).ok_or_else(|| {
                    anyhow::anyhow!("unknown template variable \"{var_name}\" in \"{s}\"")
                })?;
                segments.push(Segment::Variable(var));
            } else if ch == '}' {
                if chars.peek() == Some(&'}') {
                    literal.push('}');
                    chars.next();
                } else {
                    bail!("unmatched '}}' in template: \"{s}\"");
                }
            } else {
                literal.push(ch);
            }
        }

        if !literal.is_empty() {
            segments.push(Segment::Literal(literal));
        }

        Ok(Self { segments })
    }

    /// Expand the template using properties from the given video object.
    pub fn expand(&self, obj: &BorrowedVideoObject) -> String {
        let mut out = String::with_capacity(64);
        let det = obj.get_detection_box();
        let track = obj.get_track_box();

        for seg in &self.segments {
            match seg {
                Segment::Literal(s) => out.push_str(s),
                Segment::Variable(var) => match var {
                    TemplateVar::Namespace => out.push_str(&obj.get_namespace()),
                    TemplateVar::Label => out.push_str(&obj.get_label()),
                    TemplateVar::DrawLabel => {
                        let dl = obj.get_draw_label().unwrap_or_else(|| obj.get_label());
                        out.push_str(&dl);
                    }
                    TemplateVar::Id => {
                        let _ = write!(out, "{}", obj.get_id());
                    }
                    TemplateVar::ParentId => {
                        if let Some(pid) = obj.get_parent_id() {
                            let _ = write!(out, "{pid}");
                        }
                    }
                    TemplateVar::TrackId => {
                        if let Some(tid) = obj.get_track_id() {
                            let _ = write!(out, "{tid}");
                        }
                    }
                    TemplateVar::Confidence => {
                        if let Some(c) = obj.get_confidence() {
                            let _ = write!(out, "{c:.2}");
                        }
                    }
                    TemplateVar::DetXc => {
                        let _ = write!(out, "{:.1}", det.get_xc());
                    }
                    TemplateVar::DetYc => {
                        let _ = write!(out, "{:.1}", det.get_yc());
                    }
                    TemplateVar::DetWidth => {
                        let _ = write!(out, "{:.1}", det.get_width());
                    }
                    TemplateVar::DetHeight => {
                        let _ = write!(out, "{:.1}", det.get_height());
                    }
                    TemplateVar::DetAngle => {
                        if let Some(a) = det.get_angle() {
                            let _ = write!(out, "{a:.1}");
                        }
                    }
                    TemplateVar::TrackXc => {
                        if let Some(ref tb) = track {
                            let _ = write!(out, "{:.1}", tb.get_xc());
                        }
                    }
                    TemplateVar::TrackYc => {
                        if let Some(ref tb) = track {
                            let _ = write!(out, "{:.1}", tb.get_yc());
                        }
                    }
                    TemplateVar::TrackWidth => {
                        if let Some(ref tb) = track {
                            let _ = write!(out, "{:.1}", tb.get_width());
                        }
                    }
                    TemplateVar::TrackHeight => {
                        if let Some(ref tb) = track {
                            let _ = write!(out, "{:.1}", tb.get_height());
                        }
                    }
                    TemplateVar::TrackAngle => {
                        if let Some(ref tb) = track {
                            if let Some(a) = tb.get_angle() {
                                let _ = write!(out, "{a:.1}");
                            }
                        }
                    }
                },
            }
        }
        out
    }

    /// Returns the parsed segments (for testing / inspection).
    pub fn segments(&self) -> &[Segment] {
        &self.segments
    }
}

/// Pre-parsed multi-line label format.  Each entry corresponds to one line
/// in the rendered label.
///
/// Stores a [`format_hash`] of the source strings so consumers can detect
/// staleness with an O(1) comparison instead of re-parsing.
#[derive(Debug, Clone)]
pub struct ParsedLabelFormats {
    lines: Vec<LabelTemplate>,
    source_hash: u64,
}

impl ParsedLabelFormats {
    /// Parse a list of format strings (one per label line).
    ///
    /// The hash of `formats` is computed and stored for later comparison
    /// via [`Self::source_hash`].
    pub fn parse(formats: &[String]) -> Result<Self> {
        let lines = formats
            .iter()
            .map(|s| LabelTemplate::parse(s))
            .collect::<Result<Vec<_>>>()?;
        let source_hash = format_hash(formats);
        Ok(Self { lines, source_hash })
    }

    /// The hash of the source format strings that produced this instance.
    pub fn source_hash(&self) -> u64 {
        self.source_hash
    }

    /// Expand all lines for the given object.
    pub fn expand_lines(&self, obj: &BorrowedVideoObject) -> Vec<String> {
        self.lines.iter().map(|t| t.expand(obj)).collect()
    }

    /// Returns the number of template lines.
    pub fn line_count(&self) -> usize {
        self.lines.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_literal_only() {
        let t = LabelTemplate::parse("hello world").unwrap();
        assert_eq!(t.segments(), &[Segment::Literal("hello world".into())]);
    }

    #[test]
    fn parse_single_variable() {
        let t = LabelTemplate::parse("{namespace}").unwrap();
        assert_eq!(t.segments(), &[Segment::Variable(TemplateVar::Namespace)]);
    }

    #[test]
    fn parse_mixed() {
        let t = LabelTemplate::parse("{namespace}/{label} #{id}").unwrap();
        assert_eq!(
            t.segments(),
            &[
                Segment::Variable(TemplateVar::Namespace),
                Segment::Literal("/".into()),
                Segment::Variable(TemplateVar::Label),
                Segment::Literal(" #".into()),
                Segment::Variable(TemplateVar::Id),
            ]
        );
    }

    #[test]
    fn parse_escaped_braces() {
        let t = LabelTemplate::parse("{{literal}}").unwrap();
        assert_eq!(t.segments(), &[Segment::Literal("{literal}".into())]);
    }

    #[test]
    fn parse_unknown_variable_errors() {
        assert!(LabelTemplate::parse("{unknown_var}").is_err());
    }

    #[test]
    fn parse_unclosed_brace_errors() {
        assert!(LabelTemplate::parse("{namespace").is_err());
    }

    #[test]
    fn parse_detection_and_tracking_vars() {
        let t = LabelTemplate::parse("{det_xc},{track_xc}").unwrap();
        assert_eq!(
            t.segments(),
            &[
                Segment::Variable(TemplateVar::DetXc),
                Segment::Literal(",".into()),
                Segment::Variable(TemplateVar::TrackXc),
            ]
        );
    }

    #[test]
    fn parsed_label_formats_multi_line() {
        let formats = vec![
            "{namespace}/{label}".to_string(),
            "id={id}".to_string(),
            "conf={confidence}".to_string(),
        ];
        let plf = ParsedLabelFormats::parse(&formats).unwrap();
        assert_eq!(plf.line_count(), 3);
    }

    #[test]
    fn parse_all_variables() {
        let vars = [
            "namespace",
            "label",
            "draw_label",
            "id",
            "parent_id",
            "track_id",
            "confidence",
            "det_xc",
            "det_yc",
            "det_width",
            "det_height",
            "det_angle",
            "track_xc",
            "track_yc",
            "track_width",
            "track_height",
            "track_angle",
        ];
        for var in &vars {
            let template_str = format!("{{{var}}}");
            let t = LabelTemplate::parse(&template_str);
            assert!(t.is_ok(), "failed to parse variable: {var}");
        }
    }

    #[test]
    fn source_hash_stable_for_same_input() {
        let formats = vec!["{label}".to_string(), "{id}".to_string()];
        let a = ParsedLabelFormats::parse(&formats).unwrap();
        let b = ParsedLabelFormats::parse(&formats).unwrap();
        assert_eq!(a.source_hash(), b.source_hash());
    }

    #[test]
    fn source_hash_differs_for_different_input() {
        let a = ParsedLabelFormats::parse(&["{label}".to_string()]).unwrap();
        let b = ParsedLabelFormats::parse(&["{id}".to_string()]).unwrap();
        assert_ne!(a.source_hash(), b.source_hash());
    }

    #[test]
    fn format_hash_matches_parsed_source_hash() {
        let formats = vec!["{namespace}/{label}".to_string()];
        let plf = ParsedLabelFormats::parse(&formats).unwrap();
        assert_eq!(plf.source_hash(), format_hash(&formats));
    }
}
