use pyo3::prelude::*;

#[derive(Debug, Clone)]
enum MeterFlavor {
    CountBased(u64),
    TimeBased(u64),
}

#[derive(Debug, Clone)]
enum TimeCounter {
    #[cfg(test)]
    TestCounter(i64),
    #[cfg(not(test))]
    RealTimeCounter,
}

impl TimeCounter {
    pub fn get_time_sec(&self) -> u64 {
        match self {
            #[cfg(test)]
            TimeCounter::TestCounter(t) => u64::try_from(*t).unwrap(),
            #[cfg(not(test))]
            TimeCounter::RealTimeCounter => {
                let now = std::time::SystemTime::now();
                now.duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()
            }
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct FpsMeter {
    counter: TimeCounter,
    inner: MeterFlavor,
    frame_counter: u64,
    last_reset_frame_counter: u64,
    last_reset_time: u64,
}

impl Default for FpsMeter {
    fn default() -> Self {
        #[cfg(test)]
        let c = TimeCounter::TestCounter(0);
        #[cfg(not(test))]
        let c = TimeCounter::RealTimeCounter;
        let t = c.get_time_sec();
        Self {
            counter: c,
            inner: MeterFlavor::CountBased(0),
            frame_counter: 0,
            last_reset_frame_counter: 0,
            last_reset_time: t,
        }
    }
}

impl FpsMeter {
    pub fn get_time_sec(&self) -> u64 {
        self.counter.get_time_sec()
    }

    pub fn get_initial_value(&self) -> u64 {
        match self.inner {
            MeterFlavor::CountBased(_) => self.last_reset_frame_counter,
            MeterFlavor::TimeBased(_) => self.get_time_sec(),
        }
    }

    pub fn delta_frames(&self) -> u64 {
        self.frame_counter - self.last_reset_frame_counter
    }

    pub fn delta_time(&self) -> u64 {
        self.get_time_sec() - self.last_reset_time
    }
}

#[pymethods]
impl FpsMeter {
    #[staticmethod]
    pub fn count_based(count: i64) -> Self {
        Self {
            inner: MeterFlavor::CountBased(u64::try_from(count).unwrap()),
            ..Self::default()
        }
    }

    #[staticmethod]
    pub fn time_based(seconds: i64) -> Self {
        Self {
            inner: MeterFlavor::TimeBased(u64::try_from(seconds).unwrap()),
            ..Self::default()
        }
    }

    pub fn reset(&mut self) {
        self.last_reset_frame_counter = self.frame_counter;
        self.last_reset_time = self.get_time_sec();
    }

    pub fn __call__(&mut self, n: i64) -> (i64, i64) {
        self.frame_counter += u64::try_from(n).unwrap();
        let delta_frames = self.delta_frames();
        let delta_time = self.delta_time();
        if self.period_passed() {
            self.reset();
        }

        (
            i64::try_from(delta_frames).unwrap(),
            i64::try_from(delta_time).unwrap(),
        )
    }

    fn period_passed(&self) -> bool {
        match self.inner {
            MeterFlavor::CountBased(count) => self.delta_frames() >= count,
            MeterFlavor::TimeBased(seconds) => self.delta_time() >= seconds,
        }
    }

    #[staticmethod]
    pub fn fps(delta_frames: i64, delta_time: i64) -> Option<f64> {
        match delta_time {
            0 => None,
            _ => Some(delta_frames as f64 / delta_time as f64),
        }
    }

    #[staticmethod]
    pub fn message(delta_frames: i64, delta_time: i64) -> String {
        format!(
            "Processed {} frames in {} seconds, FPS {:?} fps",
            delta_frames,
            delta_time,
            Self::fps(delta_frames, delta_time)
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::fps_meter::TimeCounter;
    use crate::utils::FpsMeter;

    fn get_count_based_fps_meter() -> super::FpsMeter {
        super::FpsMeter::count_based(10)
    }

    fn get_time_based_fps_meter() -> super::FpsMeter {
        super::FpsMeter::time_based(5)
    }

    #[test]
    fn test_count_based_fps_meter() {
        let mut fps_meter = get_count_based_fps_meter();
        let (delta_frames, delta_time) = fps_meter.__call__(1);
        assert_eq!(delta_frames, 1);
        assert_eq!(delta_time, 0);
        let (delta_frames, delta_time) = fps_meter.__call__(1);
        assert_eq!(delta_frames, 2);
        assert_eq!(delta_time, 0);
        let (delta_frames, delta_time) = fps_meter.__call__(8);
        assert_eq!(delta_frames, 10);
        assert_eq!(delta_time, 0);
        assert_eq!(FpsMeter::fps(delta_frames, delta_time), None);

        fps_meter.counter = TimeCounter::TestCounter(1);
        let (delta_frames, delta_time) = fps_meter.__call__(1);
        assert_eq!(delta_frames, 1);
        assert_eq!(delta_time, 1);
        assert_eq!(FpsMeter::fps(delta_frames, delta_time), Some(1.0));
    }

    #[test]
    fn test_time_based_fps_meter() {
        let mut fps_meter = get_time_based_fps_meter();
        let (delta_frames, delta_time) = fps_meter.__call__(1);
        assert_eq!(delta_frames, 1);
        assert_eq!(delta_time, 0);
        let (delta_frames, delta_time) = fps_meter.__call__(1);
        assert_eq!(delta_frames, 2);
        assert_eq!(delta_time, 0);
        let (delta_frames, delta_time) = fps_meter.__call__(8);
        assert_eq!(delta_frames, 10);
        assert_eq!(delta_time, 0);
        assert_eq!(FpsMeter::fps(delta_frames, delta_time), None);

        fps_meter.counter = TimeCounter::TestCounter(1);
        let (delta_frames, delta_time) = fps_meter.__call__(1);
        assert_eq!(delta_frames, 11);
        assert_eq!(delta_time, 1);
        assert_eq!(FpsMeter::fps(delta_frames, delta_time), Some(11.0));

        fps_meter.counter = TimeCounter::TestCounter(5);
        let (delta_frames, delta_time) = fps_meter.__call__(1);
        assert_eq!(delta_frames, 12);
        assert_eq!(delta_time, 5);
        assert_eq!(FpsMeter::fps(delta_frames, delta_time), Some(2.4));

        let (delta_frames, delta_time) = fps_meter.__call__(1);
        assert_eq!(delta_frames, 1);
        assert_eq!(delta_time, 0);
    }
}
