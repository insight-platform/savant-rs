use lazy_static::lazy_static;
use parking_lot::Mutex;
use uuid::Uuid;

lazy_static! {
    static ref LAST_UUID: Mutex<Uuid> = Mutex::new(Uuid::now_v7());
}

pub fn incremental_uuid_v7() -> Uuid {
    let uuid = Uuid::now_v7();
    let timestamp = uuid.get_timestamp();
    let mut last_uuid = LAST_UUID.lock();
    if timestamp == last_uuid.get_timestamp() {
        *last_uuid = Uuid::from_u128(last_uuid.as_u128() + 1);
    } else {
        *last_uuid = uuid;
    }
    *last_uuid
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_incremental_uuid_v7() {
        for _ in 0..10000 {
            let uuid1 = incremental_uuid_v7();
            let uuid2 = incremental_uuid_v7();
            assert!(uuid2.as_u128() > uuid1.as_u128());
        }
    }
}
