use std::time::Instant;

pub struct FpsMeter {
    counter: u64,
    last_time: Instant,
}

impl Default for FpsMeter {
    fn default() -> Self {
        Self {
            counter: 0,
            last_time: Instant::now(),
        }
    }
}

impl FpsMeter {
    pub fn increment(&mut self) {
        self.counter += 1;
    }

    pub fn get_fps(&mut self) -> f64 {
        let now = Instant::now();
        let duration = now.duration_since(self.last_time);
        let secs = duration.as_secs_f64();
        let fps = if secs > 0.0 {
            self.counter as f64 / secs
        } else {
            0.0
        };
        self.last_time = now;
        self.counter = 0;
        fps
    }

    #[cfg(test)]
    pub fn set_counter(&mut self, counter: u64) {
        self.last_time = Instant::now();
        self.counter = counter;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fps_meter_basic() {
        let mut meter = FpsMeter::default();
        meter.set_counter(100);
        std::thread::sleep(std::time::Duration::from_millis(1000));
        let fps = meter.get_fps();
        assert!(
            (80.0..120.0).contains(&fps),
            "FPS was {fps} (expected rough range around 1000)"
        );
    }

    #[test]
    fn fps_meter_zero_elapsed() {
        let mut meter = FpsMeter::default();
        let fps = meter.get_fps();
        assert!(fps.is_finite());
    }
}
