use parking_lot::RwLock;
use std::sync::{Arc, Weak};

pub mod video_object_rbbox;
pub mod video_object_tracking_data;

pub type StrongInnerType<T> = Arc<RwLock<T>>;
pub type WeakInnerType<T> = Weak<RwLock<T>>;

#[derive(Clone, Debug)]
pub struct WeakInner<T> {
    inner: WeakInnerType<T>,
}

impl<T> WeakInner<T> {
    pub fn new(inner: Arc<RwLock<T>>) -> Self {
        Self {
            inner: Arc::downgrade(&inner),
        }
    }
}

pub trait UpgradeableWeakInner<T> {
    fn get_inner(&self) -> WeakInnerType<T>;

    fn get(&self) -> Option<StrongInnerType<T>> {
        self.get_inner().upgrade()
    }
    fn get_or_fail(&self) -> StrongInnerType<T> {
        self.get()
            .expect("Underlying object was dropped earlier. The reference cannot be used anymore.")
    }
}

impl<T> UpgradeableWeakInner<T> for WeakInner<T> {
    fn get_inner(&self) -> WeakInnerType<T> {
        self.inner.clone()
    }
}
