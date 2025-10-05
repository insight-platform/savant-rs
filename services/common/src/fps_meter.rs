use std::time::SystemTime;

pub struct FpsMeter {
    counter: u64,
    last_time: SystemTime,
}

impl Default for FpsMeter {
    fn default() -> Self {
        Self {
            counter: 0,
            last_time: SystemTime::now(),
        }
    }
}

impl FpsMeter {
    pub fn increment(&mut self) {
        self.counter += 1;
    }

    pub fn get_fps(&mut self) -> f64 {
        let now = SystemTime::now();
        let duration = now.duration_since(self.last_time).unwrap();
        let fps = self.counter as f64 / duration.as_secs_f64();
        self.last_time = now;
        self.counter = 0;
        fps
    }
}
