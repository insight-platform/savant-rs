use std::fmt;
use std::sync::atomic::{AtomicU32, Ordering};

pub struct AtomicF32(AtomicU32);

impl serde::Serialize for AtomicF32 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.get().serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for AtomicF32 {
    fn deserialize<D>(deserializer: D) -> Result<AtomicF32, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = f32::deserialize(deserializer)?;
        Ok(AtomicF32::new(value))
    }
}

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

impl fmt::Debug for AtomicF32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.get())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_ser_deser() {
        #[allow(clippy::approx_constant)]
        let a = super::AtomicF32::new(3.14);
        let serialized = serde_json::to_string(&a).unwrap();
        let deserialized: super::AtomicF32 = serde_json::from_str(&serialized).unwrap();
        assert_eq!(a, deserialized);
    }
}
