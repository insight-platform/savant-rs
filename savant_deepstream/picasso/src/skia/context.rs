//! Per-worker drawing context that caches expensive Skia resources
//! (typefaces, fonts, paints) and pre-parsed label templates across frames.
//!
//! Created once when a worker starts and reused for every
//! [`draw_object`](crate::skia::object::draw_object) call, avoiding
//! repeated `FontMgr::default()` + `match_family_style()` lookups and
//! template re-parsing.

use crate::spec::draw::ObjectDrawSpec;
use savant_core::draw::ObjectDraw;
use savant_core::label_template::{format_hash, ParsedLabelFormats};
use std::collections::HashMap;

/// Key for the template cache: `(namespace, label)`.
type SpecKey = (String, String);

pub struct DrawContext {
    pub default_typeface: skia_safe::Typeface,
    pub stroke_paint: skia_safe::Paint,
    pub fill_paint: skia_safe::Paint,
    font_family: String,
    template_cache: HashMap<SpecKey, ParsedLabelFormats>,
}

impl DrawContext {
    /// Create a new draw context that caches a typeface for `font_family`.
    pub fn new(font_family: &str) -> Self {
        let default_typeface = skia_safe::FontMgr::default()
            .match_family_style(font_family, skia_safe::FontStyle::default())
            .unwrap_or_else(|| {
                panic!("Failed to find font family \"{font_family}\"");
            });

        let mut stroke_paint = skia_safe::Paint::default();
        stroke_paint.set_anti_alias(true);
        stroke_paint.set_style(skia_safe::PaintStyle::Stroke);

        let mut fill_paint = skia_safe::Paint::default();
        fill_paint.set_anti_alias(true);
        fill_paint.set_style(skia_safe::PaintStyle::Fill);

        Self {
            default_typeface,
            stroke_paint,
            fill_paint,
            font_family: font_family.to_string(),
            template_cache: HashMap::new(),
        }
    }

    /// Returns the font family this context was created with.
    pub fn font_family(&self) -> &str {
        &self.font_family
    }

    /// Pre-parse label templates from the given draw spec and cache them.
    ///
    /// Should be called once when a `SourceSpec` is set or updated.
    /// Only re-parses entries whose format strings have actually changed
    /// (detected via [`ParsedLabelFormats::source_hash`]); unchanged
    /// entries are kept as-is.
    pub fn rebuild_template_cache(&mut self, spec: &ObjectDrawSpec) {
        let new_keys: HashMap<SpecKey, &[String]> = spec
            .iter()
            .filter_map(|(key, draw)| {
                draw.label
                    .as_ref()
                    .filter(|ld| !ld.format.is_empty())
                    .map(|ld| (key.clone(), ld.format.as_slice()))
            })
            .collect();

        self.template_cache.retain(|k, _| new_keys.contains_key(k));

        for (key, format) in new_keys {
            let new_hash = format_hash(format);
            if let Some(existing) = self.template_cache.get(&key) {
                if existing.source_hash() == new_hash {
                    continue;
                }
            }
            match ParsedLabelFormats::parse(format) {
                Ok(parsed) => {
                    self.template_cache.insert(key, parsed);
                }
                Err(e) => {
                    log::error!("failed to parse label template for key {:?}: {e}", key);
                }
            }
        }
    }

    /// Resolve pre-parsed templates for the given object, validating that the
    /// cached entry still matches the format strings in the provided
    /// [`ObjectDraw`].  If the draw spec was overridden by a callback and the
    /// format strings differ, the template is re-parsed and the cache updated.
    ///
    /// Returns `None` when the draw spec has no label or its format is empty.
    pub fn resolve_templates(
        &mut self,
        namespace: &str,
        label: &str,
        draw: &ObjectDraw,
    ) -> Option<&ParsedLabelFormats> {
        let format = draw.label.as_ref().map(|ld| &ld.format)?;
        if format.is_empty() {
            return None;
        }

        let key = (namespace.to_string(), label.to_string());
        let new_hash = format_hash(format);

        if let Some(existing) = self.template_cache.get(&key) {
            if existing.source_hash() == new_hash {
                return self.template_cache.get(&key);
            }
        }

        match ParsedLabelFormats::parse(format) {
            Ok(parsed) => {
                self.template_cache.insert(key.clone(), parsed);
                self.template_cache.get(&key)
            }
            Err(e) => {
                log::error!("failed to parse label template at render time: {e}");
                None
            }
        }
    }
}
