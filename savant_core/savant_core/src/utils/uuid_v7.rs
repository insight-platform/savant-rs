use std::time::Duration;

use lazy_static::lazy_static;
use parking_lot::Mutex;
use rand::Rng;
use thiserror::Error;
use uuid::{timestamp::context, Timestamp, Uuid};

/// Errors from [`relative_time_uuid_v7`].
#[derive(Debug, Error)]
pub enum RelativeTimeUuidV7Error {
    #[error(
        "UUID {uuid} has no embedded timestamp (UUID version {version}); expected v1, v6, or v7"
    )]
    NoTimestamp { uuid: Uuid, version: usize },
    #[error("timestamp overflow when adding {offset_millis} ms to UUID {uuid}")]
    TimestampAddOverflow { uuid: Uuid, offset_millis: i64 },
    #[error("timestamp underflow when subtracting {sub_ms} ms from UUID {uuid}")]
    TimestampSubUnderflow { uuid: Uuid, sub_ms: u64 },
}

lazy_static! {
    static ref LAST_UUID: Mutex<Uuid> = Mutex::new(Uuid::now_v7());
}

pub fn incremental_uuid_v7() -> Uuid {
    let uuid = Uuid::now_v7();
    let timestamp = uuid.get_timestamp();
    let mut last_uuid = LAST_UUID.lock();
    if timestamp == last_uuid.get_timestamp() {
        let mut rng = rand::rng();
        let n: u128 = rng.random_range(1..100);
        *last_uuid = Uuid::from_u128(last_uuid.as_u128() + n);
    } else {
        *last_uuid = uuid;
    }
    *last_uuid
}

pub fn relative_time_uuid_v7(
    uuid: Uuid,
    offset_millis: i64,
) -> Result<Uuid, RelativeTimeUuidV7Error> {
    let version = uuid.get_version_num();
    let ts = uuid
        .get_timestamp()
        .ok_or(RelativeTimeUuidV7Error::NoTimestamp { uuid, version })?;
    let (secs, nanos) = ts.to_unix();
    let duration = Duration::new(secs, nanos);
    let new_duration = if offset_millis > 0 {
        duration
            .checked_add(Duration::from_millis(offset_millis as u64))
            .ok_or(RelativeTimeUuidV7Error::TimestampAddOverflow {
                uuid,
                offset_millis,
            })?
    } else {
        let sub_ms = offset_millis.unsigned_abs();
        duration
            .checked_sub(Duration::from_millis(sub_ms))
            .ok_or(RelativeTimeUuidV7Error::TimestampSubUnderflow { uuid, sub_ms })?
    };
    Ok(Uuid::new_v7(Timestamp::from_unix(
        context::ContextV7::new(),
        new_duration.as_secs(),
        new_duration.subsec_nanos(),
    )))
}

#[cfg(test)]
mod tests {
    use std::{thread::sleep, time::Duration};

    use super::*;

    #[test]
    fn test_incremental_uuid_v7() {
        for _ in 0..10000 {
            let uuid1 = incremental_uuid_v7();
            let uuid2 = incremental_uuid_v7();
            assert!(uuid2.as_u128() > uuid1.as_u128());
        }
    }

    #[test]
    fn test_relative_time_uuid_v7() {
        for _ in 0..100 {
            let now_uuid = incremental_uuid_v7();
            let future_uuid = relative_time_uuid_v7(now_uuid, 1).unwrap();
            let very_past_uuid = relative_time_uuid_v7(now_uuid, -10).unwrap();
            let past_uuid = relative_time_uuid_v7(now_uuid, -1).unwrap();
            assert!(very_past_uuid.as_u128() < past_uuid.as_u128());
            assert!(past_uuid.as_u128() < now_uuid.as_u128());
            assert!(now_uuid.as_u128() < future_uuid.as_u128());
            sleep(Duration::from_millis(2));
            let now_uuid2 = incremental_uuid_v7();
            assert!(now_uuid2.as_u128() > future_uuid.as_u128());
        }
    }

    #[test]
    fn test_relative_time_uuid_v7_rejects_non_timestamp_uuid() {
        let v4 = Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").unwrap();
        let err = relative_time_uuid_v7(v4, 0).unwrap_err();
        assert!(matches!(err, RelativeTimeUuidV7Error::NoTimestamp { .. }));
    }

    #[test]
    fn test_relative_time_uuid_v7_underflow() {
        let now_uuid = incremental_uuid_v7();
        let err = relative_time_uuid_v7(now_uuid, -1_000_000_000_000_000).unwrap_err();
        assert!(matches!(
            err,
            RelativeTimeUuidV7Error::TimestampSubUnderflow { .. }
        ));
    }
}
