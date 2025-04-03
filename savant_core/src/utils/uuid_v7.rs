use std::time::Duration;

use lazy_static::lazy_static;
use parking_lot::Mutex;
use rand::Rng;
use uuid::{timestamp::context, Timestamp, Uuid};

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

pub fn relative_time_uuid_v7(uuid: Uuid, offset_millis: i64) -> Uuid {
    let ts = uuid.get_timestamp().unwrap();
    let (secs, nanos) = ts.to_unix();
    let duration = Duration::new(secs, nanos);
    let new_duration = if offset_millis > 0 {
        duration + Duration::from_millis(offset_millis as u64)
    } else {
        duration - Duration::from_millis(offset_millis.abs() as u64)
    };
    let new_uuid = Uuid::new_v7(Timestamp::from_unix(
        context::ContextV7::new(),
        new_duration.as_secs(),
        new_duration.subsec_nanos(),
    ));
    new_uuid
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
        let now_uuid = incremental_uuid_v7();
        let future_uuid = relative_time_uuid_v7(now_uuid, 1);
        let very_past_uuid = relative_time_uuid_v7(now_uuid, -10);
        let past_uuid = relative_time_uuid_v7(now_uuid, -1);
        assert!(very_past_uuid.as_u128() < past_uuid.as_u128());
        assert!(past_uuid.as_u128() < now_uuid.as_u128());
        assert!(now_uuid.as_u128() < future_uuid.as_u128());
        sleep(Duration::from_millis(2));
        let now_uuid2 = incremental_uuid_v7();
        assert!(now_uuid2.as_u128() > future_uuid.as_u128());
    }
}
