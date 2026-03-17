use hashbrown::{Equivalent, HashMap};
use savant_core::draw::ObjectDraw;
use std::hash::{Hash, Hasher};

/// Borrowed pair of `&str` that can look up a `(String, String)` key in a
/// [`hashbrown::HashMap`] without allocating.
///
/// The [`Hash`] implementation is compatible with `(String, String)` because
/// the standard tuple hash simply hashes each element in order, and
/// `str` / `String` produce identical hashes.
pub(crate) struct StrPairKey<'a>(pub &'a str, pub &'a str);

impl Hash for StrPairKey<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
        self.1.hash(state);
    }
}

impl Equivalent<(String, String)> for StrPairKey<'_> {
    fn equivalent(&self, key: &(String, String)) -> bool {
        self.0 == key.0 && self.1 == key.1
    }
}

/// Static per-object draw specifications keyed by `(namespace, label)`.
///
/// Lookup is strict — only exact matches are returned.
#[derive(Debug, Clone, Default)]
pub struct ObjectDrawSpec {
    map: HashMap<(String, String), ObjectDraw>,
}

impl ObjectDrawSpec {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a draw spec for the given `(namespace, label)` key.
    pub fn insert(&mut self, namespace: &str, label: &str, draw: ObjectDraw) {
        self.map
            .insert((namespace.to_string(), label.to_string()), draw);
    }

    /// Look up the `ObjectDraw` for the exact `(namespace, label)` pair.
    ///
    /// Uses [`StrPairKey`] to avoid heap-allocating two `String`s on every
    /// call — this is the hot path (called per object per frame).
    pub fn lookup(&self, namespace: &str, label: &str) -> Option<&ObjectDraw> {
        self.map.get(&StrPairKey(namespace, label))
    }

    /// Iterate over all `(key, ObjectDraw)` entries.
    pub fn iter(&self) -> impl Iterator<Item = (&(String, String), &ObjectDraw)> {
        self.map.iter()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use savant_core::draw::ObjectDraw;

    fn make_draw(blur: bool) -> ObjectDraw {
        ObjectDraw::new(None, None, None, blur)
    }

    #[test]
    fn empty_lookup_returns_none() {
        let spec = ObjectDrawSpec::new();
        assert!(spec.lookup("ns", "lbl").is_none());
    }

    #[test]
    fn exact_match() {
        let mut spec = ObjectDrawSpec::new();
        spec.insert("det", "car", make_draw(true));
        let found = spec.lookup("det", "car").unwrap();
        assert!(found.blur);
    }

    #[test]
    fn no_cross_match() {
        let mut spec = ObjectDrawSpec::new();
        spec.insert("det", "car", make_draw(true));
        assert!(spec.lookup("det", "truck").is_none());
        assert!(spec.lookup("other", "car").is_none());
    }

    #[test]
    fn multiple_entries() {
        let mut spec = ObjectDrawSpec::new();
        spec.insert("det", "car", make_draw(true));
        spec.insert("det", "person", make_draw(false));
        assert!(spec.lookup("det", "car").unwrap().blur);
        assert!(!spec.lookup("det", "person").unwrap().blur);
    }
}
