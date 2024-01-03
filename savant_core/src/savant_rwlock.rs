use rkyv::with::{ArchiveWith, DeserializeWith, Immutable, Lock, SerializeWith};
use rkyv::{Archive, Deserialize, Fallible, Serialize};
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
    pub fn clone(&self) -> Self {
        Self(self.0.clone())
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

impl<F: Archive> ArchiveWith<SavantRwLock<F>> for Lock {
    type Archived = Immutable<F::Archived>;
    type Resolver = F::Resolver;

    #[inline]
    unsafe fn resolve_with(
        field: &SavantRwLock<F>,
        pos: usize,
        resolver: Self::Resolver,
        out: *mut Self::Archived,
    ) {
        field.read_recursive().resolve(pos, resolver, out.cast());
    }
}

impl<F: Serialize<S>, S: Fallible + ?Sized> SerializeWith<SavantRwLock<F>, S> for Lock {
    #[inline]
    fn serialize_with(
        field: &SavantRwLock<F>,
        serializer: &mut S,
    ) -> Result<Self::Resolver, S::Error> {
        field.read_recursive().serialize(serializer)
    }
}

impl<F, T, D> DeserializeWith<Immutable<F>, SavantRwLock<T>, D> for Lock
where
    F: Deserialize<T, D>,
    D: Fallible + ?Sized,
{
    #[inline]
    fn deserialize_with(
        field: &Immutable<F>,
        deserializer: &mut D,
    ) -> Result<SavantRwLock<T>, D::Error> {
        Ok(SavantRwLock::new(field.value().deserialize(deserializer)?))
    }
}

impl<F: Archive> ArchiveWith<SavantArcRwLock<F>> for Lock {
    type Archived = Immutable<F::Archived>;
    type Resolver = F::Resolver;

    #[inline]
    unsafe fn resolve_with(
        field: &SavantArcRwLock<F>,
        pos: usize,
        resolver: Self::Resolver,
        out: *mut Self::Archived,
    ) {
        field.0.read_recursive().resolve(pos, resolver, out.cast());
    }
}

impl<F: Serialize<S>, S: Fallible + ?Sized> SerializeWith<SavantArcRwLock<F>, S> for Lock {
    #[inline]
    fn serialize_with(
        field: &SavantArcRwLock<F>,
        serializer: &mut S,
    ) -> Result<Self::Resolver, S::Error> {
        field.0.read_recursive().serialize(serializer)
    }
}

impl<F, T, D> DeserializeWith<Immutable<F>, SavantArcRwLock<T>, D> for Lock
where
    F: Deserialize<T, D>,
    D: Fallible + ?Sized,
{
    #[inline]
    fn deserialize_with(
        field: &Immutable<F>,
        deserializer: &mut D,
    ) -> Result<SavantArcRwLock<T>, D::Error> {
        Ok(SavantArcRwLock::new(
            field.value().deserialize(deserializer)?,
        ))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test() {}
}
