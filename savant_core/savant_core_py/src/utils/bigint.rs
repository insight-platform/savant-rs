use num_bigint::BigInt;

/// Returned when a [`BigInt`] cannot be represented as [`i64`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FitI64Error {
    pub value: BigInt,
}

impl std::fmt::Display for FitI64Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "value {} does not fit in i64 range (valid range: {}..={})",
            self.value,
            i64::MIN,
            i64::MAX
        )
    }
}

impl std::error::Error for FitI64Error {}

/// Convert `v` to `i64` if it lies in [`i64::MIN`, `i64::MAX`].
pub fn fit_i64(v: BigInt) -> Result<i64, FitI64Error> {
    match i64::try_from(&v) {
        Ok(i) => Ok(i),
        Err(_) => Err(FitI64Error { value: v }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fit_i64_in_range() {
        assert_eq!(fit_i64(BigInt::from(i64::MAX)).unwrap(), i64::MAX);
        assert_eq!(fit_i64(BigInt::from(i64::MIN)).unwrap(), i64::MIN);
        assert_eq!(fit_i64(BigInt::from(0)).unwrap(), 0);
    }

    #[test]
    fn test_fit_i64_out_of_range() {
        assert!(fit_i64(BigInt::from(i64::MAX) + 1).is_err());
        assert!(fit_i64(BigInt::from(i64::MIN) - 1).is_err());
    }
}
