use rkyv::{with::Atomic, Archive, Deserialize, Serialize};
use std::sync::atomic::{AtomicU32, Ordering};

#[derive(Archive, Deserialize, Serialize, Debug, serde::Serialize, serde::Deserialize)]
#[archive(check_bytes)]
pub struct AtomicF32(#[with(Atomic)] AtomicU32);

impl PartialEq for AtomicF32 {
    fn eq(&self, other: &Self) -> bool {
        self.get() == other.get()
    }
}

impl Clone for AtomicF32 {
    fn clone(&self) -> Self {
        Self::new(self.get())
    }
}

impl AtomicF32 {
    pub fn new(value: f32) -> Self {
        let as_u32 = value.to_bits();
        Self(AtomicU32::new(as_u32))
    }
    pub fn set(&self, value: f32) {
        let as_u32 = value.to_bits();
        self.0.store(as_u32, Ordering::SeqCst)
    }
    pub fn get(&self) -> f32 {
        let as_u32 = self.0.load(Ordering::SeqCst);
        f32::from_bits(as_u32)
    }
}

impl From<f32> for AtomicF32 {
    fn from(value: f32) -> Self {
        Self::new(value)
    }
}

impl From<AtomicF32> for f32 {
    fn from(value: AtomicF32) -> Self {
        value.get()
    }
}
