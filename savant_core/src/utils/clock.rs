pub mod mock_clock {
    use std::{
        cell::RefCell,
        time::{Duration, SystemTime},
    };

    pub struct MockClock {
        current_time: u64,
    }

    impl MockClock {
        pub fn new() -> Self {
            Self { current_time: 0 }
        }

        pub fn now(&self) -> SystemTime {
            SystemTime::UNIX_EPOCH + Duration::from_millis(self.current_time)
        }

        pub fn reset_time(&mut self) {
            self.current_time = 0;
        }

        pub fn advance_time_ms(&mut self, d: u64) {
            self.current_time += d;
        }
    }

    thread_local! {
        static CLOCK: RefCell<MockClock> = RefCell::new(MockClock::new());
    }

    pub fn now() -> SystemTime {
        CLOCK.with(|clock| clock.borrow().now())
    }

    pub fn reset_time() {
        CLOCK.with(|clock| clock.borrow_mut().reset_time());
    }

    pub fn advance_time_ms(d: u64) {
        CLOCK.with(|clock| clock.borrow_mut().advance_time_ms(d));
    }
}

pub mod clock {
    use std::time::SystemTime;

    pub fn now() -> SystemTime {
        SystemTime::now()
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, SystemTime};

    use super::*;

    #[test]
    fn test_mock_clock() {
        let mut clock = mock_clock::MockClock::new();
        assert_eq!(clock.now(), SystemTime::UNIX_EPOCH);
        clock.advance_time_ms(1000);
        assert_eq!(
            clock.now(),
            SystemTime::UNIX_EPOCH + Duration::from_millis(1000)
        );
    }
}
