use parking_lot::Mutex;
use rkyv::{with::Skip, Archive, Deserialize, Serialize};
use std::any::Any;
use std::sync::Arc;

#[derive(Archive, Deserialize, Serialize, Debug, Clone, serde::Serialize, serde::Deserialize)]
#[archive(check_bytes)]
pub struct AnyObject {
    #[with(Skip)]
    #[serde(skip_deserializing, skip_serializing)]
    pub value: Arc<Mutex<Option<Box<dyn Any + Send>>>>,
}

impl Default for AnyObject {
    fn default() -> Self {
        Self {
            value: Arc::new(Mutex::new(None)),
        }
    }
}

impl PartialEq for AnyObject {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

impl AnyObject {
    pub fn new(value: Box<dyn Any + Send>) -> Self {
        Self {
            value: Arc::new(Mutex::new(Some(value))),
        }
    }
    pub fn set(&self, value: Box<dyn Any + Send>) {
        let mut bind = self.value.lock();
        *bind = Some(value);
    }

    pub fn take(&self) -> Option<Box<dyn Any + Send>> {
        let mut value = self.value.lock();
        value.take()
    }

    pub fn access(&self) -> Arc<Mutex<Option<Box<dyn Any + Send>>>> {
        self.value.clone()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_any_object() {
        let p = super::AnyObject::new(Box::new(1.0));
        let v = p.take().unwrap();
        let v = v.downcast::<f64>().unwrap();
        assert_eq!(v, Box::new(1.0));
    }
}
