//! [`Registry`] — the framework's name-keyed directory of actor
//! addresses.
//!
//! Every actor / source published during
//! [`System::build`](super::actor::Actor) reserves its
//! [`StageName`] slot and writes the corresponding `Addr<M>` into
//! this registry.  Peers look up typed addresses during
//! construction (`BuildCtx::addr::<M>(&peer)`) or at runtime
//! (`Context::resolve::<M>(&peer)`); routing is name-based, not
//! position-based, so a pipeline can add / swap / reroute actors
//! without rewiring shared-sender boilerplate.
//!
//! The registry is populated in phase 1 of `System::build` and
//! frozen from phase 2 onwards — all lookup methods take `&self`,
//! so the registry is safely shareable as `Arc<Registry>` across
//! every actor thread.

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{anyhow, Result};

use super::addr::Addr;
use super::envelope::Envelope;
use super::supervisor::{StageKind, StageName};

type Entry = Arc<dyn Any + Send + Sync>;

/// Name-keyed directory of typed actor addresses.
///
/// Entries are type-erased behind `Arc<dyn Any + Send + Sync>` so
/// one registry can hold addresses of *any* envelope type;
/// [`Registry::get::<M>`] downcasts back to the caller's expected
/// `Addr<M>` and returns an error if the stored entry has a
/// different envelope type.
#[derive(Default)]
pub struct Registry {
    entries: HashMap<StageName, Entry>,
}

impl Registry {
    /// Empty registry — populated by `System::build` phase 1.
    pub fn new() -> Self {
        Self::default()
    }

    /// Publish `addr` under `name`.  Panics if `name` is already
    /// taken: double-registration is always a programming bug, so
    /// we prefer the loud failure here over returning an error.
    pub fn insert<M: Envelope>(&mut self, name: StageName, addr: Addr<M>) {
        use std::collections::hash_map::Entry as MapEntry;
        match self.entries.entry(name.clone()) {
            MapEntry::Occupied(_) => {
                panic!("registry: duplicate stage name {name}");
            }
            MapEntry::Vacant(v) => {
                v.insert(Arc::new(addr) as Entry);
            }
        }
    }

    /// Look up the address published under `name` as `Addr<M>`.
    ///
    /// Returns `Err` if either `name` has no registered entry or
    /// the registered entry is for a different envelope type.
    pub fn get<M: Envelope>(&self, name: &StageName) -> Result<Addr<M>> {
        let entry = self
            .entries
            .get(name)
            .ok_or_else(|| anyhow!("registry: no stage named {name}"))?;
        entry
            .clone()
            .downcast::<Addr<M>>()
            .map(|arc| (*arc).clone())
            .map_err(|_| {
                anyhow!(
                    "registry: stage {name} is registered with a different message type \
                     (expected {})",
                    std::any::type_name::<M>()
                )
            })
    }

    /// Whether `name` has a registered entry.
    pub fn contains(&self, name: &StageName) -> bool {
        self.entries.contains_key(name)
    }

    /// Snapshot of every registered [`StageName`], in insertion-
    /// order-independent form.  Intended for introspection and
    /// debug logging.
    pub fn stages(&self) -> Vec<StageName> {
        self.entries.keys().cloned().collect()
    }

    /// Registered [`StageName`]s whose
    /// [`kind`](StageName::kind) matches `kind`.
    pub fn stages_of(&self, kind: StageKind) -> Vec<StageName> {
        self.entries
            .keys()
            .filter(|n| n.kind == kind)
            .cloned()
            .collect()
    }
}

impl std::fmt::Debug for Registry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut dbg = f.debug_struct("Registry");
        for name in self.entries.keys() {
            dbg.field("stage", &format_args!("{name}"));
        }
        dbg.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framework::envelope::ShutdownHint;
    use crossbeam::channel::bounded;

    struct MsgA;
    struct MsgB;
    impl Envelope for MsgA {
        fn as_shutdown(&self) -> Option<ShutdownHint<'_>> {
            None
        }
    }
    impl Envelope for MsgB {
        fn as_shutdown(&self) -> Option<ShutdownHint<'_>> {
            None
        }
    }

    fn addr<M: Envelope>(name: StageName) -> Addr<M> {
        let (tx, _rx) = bounded::<M>(1);
        Addr::new(name, tx)
    }

    #[test]
    fn insert_and_get_round_trip() {
        let mut r = Registry::new();
        let name = StageName::unnamed(StageKind::Infer);
        r.insert::<MsgA>(name.clone(), addr(name.clone()));
        let got: Addr<MsgA> = r.get(&name).unwrap();
        assert_eq!(got.name(), &name);
    }

    #[test]
    fn get_returns_err_for_unknown_stage() {
        let r = Registry::new();
        let err = r
            .get::<MsgA>(&StageName::unnamed(StageKind::Infer))
            .unwrap_err();
        assert!(err.to_string().contains("no stage"));
    }

    #[test]
    fn get_returns_err_on_type_mismatch() {
        let mut r = Registry::new();
        let name = StageName::unnamed(StageKind::Infer);
        r.insert::<MsgA>(name.clone(), addr(name.clone()));
        let err = r.get::<MsgB>(&name).unwrap_err();
        assert!(err.to_string().contains("different message type"));
    }

    #[test]
    #[should_panic(expected = "duplicate stage name")]
    fn duplicate_insert_panics() {
        let mut r = Registry::new();
        let name = StageName::unnamed(StageKind::Infer);
        r.insert::<MsgA>(name.clone(), addr(name.clone()));
        r.insert::<MsgA>(name.clone(), addr(name));
    }

    #[test]
    fn stages_and_stages_of_filter_by_kind() {
        let mut r = Registry::new();
        r.insert::<MsgA>(
            StageName::new(StageKind::Infer, "yolo"),
            addr(StageName::new(StageKind::Infer, "yolo")),
        );
        r.insert::<MsgA>(
            StageName::new(StageKind::Infer, "attr"),
            addr(StageName::new(StageKind::Infer, "attr")),
        );
        r.insert::<MsgB>(
            StageName::unnamed(StageKind::Tracker),
            addr(StageName::unnamed(StageKind::Tracker)),
        );
        assert_eq!(r.stages().len(), 3);
        assert_eq!(r.stages_of(StageKind::Infer).len(), 2);
        assert_eq!(r.stages_of(StageKind::Tracker).len(), 1);
        assert_eq!(r.stages_of(StageKind::Picasso).len(), 0);
    }
}
