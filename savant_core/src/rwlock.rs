use std::sync::Arc;

#[derive(Debug, Default)]
pub struct SavantRwLock<T>(parking_lot::RwLock<T>);

#[derive(Debug, Default, Clone)]
pub struct SavantArcRwLock<T>(pub Arc<SavantRwLock<T>>);

impl<T> From<Arc<SavantRwLock<T>>> for SavantArcRwLock<T> {
    #[inline]
    fn from(arc: Arc<SavantRwLock<T>>) -> Self {
        Self(arc)
    }
}

impl<T> From<&Arc<SavantRwLock<T>>> for SavantArcRwLock<T> {
    #[inline]
    fn from(arc: &Arc<SavantRwLock<T>>) -> Self {
        Self(arc.clone())
    }
}

impl<T> SavantArcRwLock<T> {
    #[inline]
    pub fn new(v: T) -> Self {
        Self(Arc::new(SavantRwLock::new(v)))
    }

    #[inline]
    pub fn read(&self) -> parking_lot::RwLockReadGuard<'_, T> {
        self.0.read()
    }

    #[inline]
    pub fn read_recursive(&self) -> parking_lot::RwLockReadGuard<'_, T> {
        self.0.read_recursive()
    }

    #[inline]
    pub fn write(&self) -> parking_lot::RwLockWriteGuard<'_, T> {
        self.0.write()
    }

    // #[inline]
    // pub fn into_inner(self) -> T {
    //     let inner = self.0;
    //     inner.into_inner()
    // }
}

impl<T> SavantRwLock<T> {
    #[inline]
    pub fn new(t: T) -> Self {
        Self(parking_lot::RwLock::new(t))
    }

    #[inline]
    pub fn read(&self) -> parking_lot::RwLockReadGuard<'_, T> {
        self.0.read()
    }

    #[inline]
    pub fn read_recursive(&self) -> parking_lot::RwLockReadGuard<'_, T> {
        self.0.read_recursive()
    }

    #[inline]
    pub fn write(&self) -> parking_lot::RwLockWriteGuard<'_, T> {
        self.0.write()
    }

    #[inline]
    pub fn into_inner(self) -> T {
        self.0.into_inner()
    }
}
