use lazy_static::lazy_static;
use num_bigint::BigInt;

lazy_static! {
    // Initialize these values only once at runtime
    static ref MAX_I64: BigInt = BigInt::from(i64::MAX);
    static ref MIN_I64: BigInt = BigInt::from(i64::MIN);
    static ref MAX_U64: BigInt = BigInt::from(u64::MAX);
    static ref MIN_U64: BigInt = BigInt::from(u64::MIN);
}

pub fn fit_i64(v: BigInt) -> i64 {
    // Use the static references instead of creating new instances
    let v = if v > *MAX_I64 {
        let res = v.clone() % &*MAX_I64;
        log::warn!(
            "v is greater than i64::MAX, cropping to fit the i64 range, original: {}, result: {}",
            &v,
            &res
        );
        res
    } else if v < *MIN_I64 {
        let res = v.clone() % &*MIN_I64;
        log::warn!(
            "v is less than i64::MIN, cropping to fit the i64 range, original: {}, result: {}",
            &v,
            &res
        );
        res
    } else {
        v
    };
    i64::try_from(v).expect("v must be in the range of i64")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fit_i64() {
        assert_eq!(fit_i64(BigInt::from(i64::MAX)), i64::MAX);
        assert_eq!(fit_i64(BigInt::from(i64::MIN)), i64::MIN);
        assert_eq!(fit_i64(BigInt::from(i64::MAX) + 1), 1);
        assert_eq!(fit_i64(BigInt::from(i64::MIN) - 1), -1);
    }
}
