use std::fmt::Debug;
use std::sync::OnceLock;

#[derive(Debug, Clone)]
pub struct DefaultOnceCell<T: Clone + Debug> {
    cell: OnceLock<T>,
    default: T,
}

impl<T> DefaultOnceCell<T>
where
    T: Clone + Debug,
{
    pub fn new(default: T) -> Self {
        Self {
            cell: OnceLock::new(),
            default,
        }
    }

    pub fn set(&self, value: T) -> anyhow::Result<()> {
        self.cell
            .set(value)
            .map_err(|_| anyhow::anyhow!("Cell already initialized"))?;
        Ok(())
    }

    pub fn is_initialized(&self) -> bool {
        self.cell.get().is_some()
    }

    pub fn get_or_init(&self) -> &T {
        self.cell.get_or_init(|| self.default.clone())
    }

    pub fn get_default(&self) -> T {
        self.default.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_once_cell() {
        let cell = DefaultOnceCell::new(1);
        assert_eq!(*cell.get_or_init(), 1);
        assert!(cell.set(2).is_err());
        assert_eq!(*cell.get_or_init(), 1);
    }

    #[test]
    fn test_set_once_cell() {
        let cell = DefaultOnceCell::new(1);
        assert!(cell.set(2).is_ok());
        assert_eq!(*cell.get_or_init(), 2);
    }
}
