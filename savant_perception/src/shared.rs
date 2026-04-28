//! [`SharedStore`] — a type-keyed (and optionally name-keyed)
//! container of shared, read-only-by-default, thread-safe values.
//!
//! Use it to publish cross-actor state through
//! [`System::insert_shared`](super::system::System::insert_shared)
//! and friends, then read it from any factory or running actor:
//!
//! * **Singleton-by-type** — one value per type. Insert with
//!   [`SharedStore::insert`], look up with [`SharedStore::get::<T>`] /
//!   [`BuildCtx::shared::<T>`](super::context::BuildCtx::shared) /
//!   [`Context::shared::<T>`](super::context::Context::shared).
//! * **Keyed-by-name** — several distinct instances of the same
//!   type, scoped by an arbitrary key string. Insert with
//!   [`SharedStore::insert_as`], look up with
//!   [`SharedStore::get_as`] /
//!   [`BuildCtx::shared_as`](super::context::BuildCtx::shared_as) /
//!   [`Context::shared_as`](super::context::Context::shared_as).
//!
//! The store is mutated only through the inserter methods, which
//! all take `&mut self`. Once the store is shared as
//! `Arc<SharedStore>` (i.e. once [`System::run`](super::system::System::run)
//! has started), the public API observed by actors is read-only.
//! Mutation visible to running actors must therefore use
//! **interior mutability** in the stored `T` itself (e.g.
//! `Arc<Mutex<State>>` or atomics).  All look-up methods return a
//! cheap `Arc<T>` clone.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::Arc;

type Entry = Arc<dyn Any + Send + Sync>;

/// Type- and name-keyed store of shared state published through
/// [`System`](super::system::System). See the module docs for
/// usage patterns.
#[derive(Default)]
pub struct SharedStore {
    by_type: HashMap<TypeId, Entry>,
    by_name: HashMap<(TypeId, String), Entry>,
}

impl SharedStore {
    /// Empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Publish `value` under `TypeId::of::<T>()`.  Overrides any
    /// prior singleton of the same type — later `insert`s win,
    /// matching "the last configuration survives" semantics that
    /// are easy to reason about during pipeline composition.
    ///
    /// Wraps `value` in an `Arc<T>` internally so look-ups hand
    /// out cheap clones without reallocating.  If you already have
    /// an `Arc<T>`, call [`SharedStore::insert_arc`] to avoid the
    /// extra wrap.
    pub fn insert<T: Send + Sync + 'static>(&mut self, value: T) {
        self.by_type
            .insert(TypeId::of::<T>(), Arc::new(value) as Entry);
    }

    /// Publish an already-wrapped `Arc<T>` as the singleton of `T`.
    pub fn insert_arc<T: Send + Sync + 'static>(&mut self, value: Arc<T>) {
        self.by_type.insert(TypeId::of::<T>(), value as Entry);
    }

    /// Publish `value` under `(TypeId::of::<T>(), key)`.  Two
    /// different `T`s with the same `key` string coexist — keys
    /// are scoped by type.
    pub fn insert_as<T: Send + Sync + 'static>(&mut self, key: impl Into<String>, value: T) {
        self.by_name
            .insert((TypeId::of::<T>(), key.into()), Arc::new(value) as Entry);
    }

    /// Publish an already-wrapped `Arc<T>` under `(TypeId::of::<T>(), key)`.
    pub fn insert_arc_as<T: Send + Sync + 'static>(
        &mut self,
        key: impl Into<String>,
        value: Arc<T>,
    ) {
        self.by_name
            .insert((TypeId::of::<T>(), key.into()), value as Entry);
    }

    /// Look up the singleton of `T`, returning `Some(Arc<T>)` if
    /// present or `None` otherwise.
    pub fn get<T: Send + Sync + 'static>(&self) -> Option<Arc<T>> {
        self.by_type
            .get(&TypeId::of::<T>())
            .cloned()
            .and_then(|e| e.downcast::<T>().ok())
    }

    /// Look up the named instance of `T` keyed by `key`.
    pub fn get_as<T: Send + Sync + 'static>(&self, key: &str) -> Option<Arc<T>> {
        self.by_name
            .get(&(TypeId::of::<T>(), key.to_string()))
            .cloned()
            .and_then(|e| e.downcast::<T>().ok())
    }

    /// Whether a singleton of `T` has been published.
    pub fn contains<T: Send + Sync + 'static>(&self) -> bool {
        self.by_type.contains_key(&TypeId::of::<T>())
    }

    /// Whether a named instance of `T` keyed by `key` has been
    /// published.
    pub fn contains_as<T: Send + Sync + 'static>(&self, key: &str) -> bool {
        self.by_name
            .contains_key(&(TypeId::of::<T>(), key.to_string()))
    }
}

impl std::fmt::Debug for SharedStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedStore")
            .field("types", &self.by_type.len())
            .field("named", &self.by_name.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn singleton_round_trip() {
        let mut s = SharedStore::new();
        s.insert::<u32>(42);
        let got = s.get::<u32>().expect("must find u32");
        assert_eq!(*got, 42);
        assert!(s.contains::<u32>());
        assert!(!s.contains::<i64>());
        assert!(s.get::<i64>().is_none());
    }

    #[test]
    fn later_insert_replaces_earlier_singleton() {
        let mut s = SharedStore::new();
        s.insert::<u32>(1);
        s.insert::<u32>(2);
        assert_eq!(*s.get::<u32>().unwrap(), 2);
    }

    #[test]
    fn named_entries_scope_by_type() {
        let mut s = SharedStore::new();
        s.insert_as::<u32>("a", 1);
        s.insert_as::<u64>("a", 2);
        assert_eq!(*s.get_as::<u32>("a").unwrap(), 1);
        assert_eq!(*s.get_as::<u64>("a").unwrap(), 2);
        assert!(s.get_as::<u32>("b").is_none());
        assert!(s.contains_as::<u32>("a"));
        assert!(!s.contains_as::<u32>("b"));
    }

    #[test]
    fn insert_arc_and_insert_arc_as_preserve_shared_arc() {
        let mut s = SharedStore::new();
        let shared = Arc::new(String::from("hello"));
        s.insert_arc(shared.clone());
        let fetched = s.get::<String>().unwrap();
        assert!(Arc::ptr_eq(&shared, &fetched));

        let shared2 = Arc::new(vec![1_u32, 2, 3]);
        s.insert_arc_as("seed", shared2.clone());
        let fetched2 = s.get_as::<Vec<u32>>("seed").unwrap();
        assert!(Arc::ptr_eq(&shared2, &fetched2));
    }
}
