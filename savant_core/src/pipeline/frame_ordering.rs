use anyhow::{bail, Result};
use hashbrown::HashMap;
use parking_lot::RwLock;
use std::sync::atomic::{AtomicI64, Ordering};

#[derive(Debug)]
pub struct FrameOrdering {
    last: AtomicI64,
    order: RwLock<HashMap<i64, i64>>,
}

impl FrameOrdering {
    pub fn new() -> Self {
        Self {
            last: AtomicI64::new(-1),
            order: RwLock::new(HashMap::new()),
        }
    }

    pub fn get(&self, frame_id: i64) -> Result<Option<i64>> {
        let bind = self.order.read();
        match bind.get(&frame_id) {
            Some(v) if *v == -1 => Ok(None),
            Some(v) => Ok(Some(*v)),
            None => bail!("Frame {} not found", frame_id),
        }
    }

    pub fn add(&self, frame_id: i64) {
        let last = self
            .last
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |_| Some(frame_id))
            .unwrap();
        self.order.write().insert(frame_id, last);
    }

    pub fn delete(&self, frame_id: i64) -> Result<i64> {
        self.order.write().remove(&frame_id).ok_or(anyhow::anyhow!(
            "Frame {} not found in the ordering",
            frame_id
        ))
    }
}

#[cfg(test)]
mod tests {
    use crate::pipeline::frame_ordering::FrameOrdering;

    #[test]
    fn test_operations() {
        let fo = FrameOrdering::new();
        fo.add(1);
        assert!(matches!(fo.get(1), Ok(None)));
        fo.add(3);
        assert!(matches!(fo.get(3), Ok(Some(1))));
        fo.add(2);
        assert!(matches!(fo.get(2), Ok(Some(3))));
        assert!(matches!(fo.delete(3), Ok(v) if v == 1));
        assert!(matches!(fo.delete(3), Err(_)));
        assert!(matches!(fo.get(3), Err(_)));
    }
}
