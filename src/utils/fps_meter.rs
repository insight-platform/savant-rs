use pyo3::prelude::*;

#[derive(Debug, Clone)]
enum MeterFlavor {
    CountBased(u64),
    TimeBased(u64),
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct FpsMeter {
    inner: MeterFlavor,
    frame_counter: u64,
    last_reset_frame_counter: u64,
    last_reset_time: u64,
}

impl Default for FpsMeter {
    fn default() -> Self {
        Self {
            inner: MeterFlavor::CountBased(0),
            frame_counter: 0,
            last_reset_frame_counter: 0,
            last_reset_time: Self::get_time_sec(),
        }
    }
}

impl FpsMeter {
    pub fn get_time_sec() -> u64 {
        let now = std::time::SystemTime::now();
        now.duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()
    }

    pub fn get_initial_value(&self) -> u64 {
        match self.inner {
            MeterFlavor::CountBased(_) => self.last_reset_frame_counter,
            MeterFlavor::TimeBased(_) => Self::get_time_sec(),
        }
    }

    pub fn delta_frames(&self) -> u64 {
        self.frame_counter - self.last_reset_frame_counter
    }

    pub fn delta_time(&self) -> u64 {
        Self::get_time_sec() - self.last_reset_time
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
        self.last_reset_time = Self::get_time_sec();
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
            i64::try_from(delta_time as i64).unwrap(),
        )
    }

    fn period_passed(&self) -> bool {
        match self.inner {
            MeterFlavor::CountBased(count) => self.delta_frames() >= count,
            MeterFlavor::TimeBased(seconds) => self.delta_time() >= seconds,
        }
    }

    #[staticmethod]
    pub fn fps(delta_frames: i64, delta_time: i64) -> f64 {
        delta_frames as f64 / delta_time as f64
    }

    #[staticmethod]
    pub fn message(delta_frames: i64, delta_time: i64) -> String {
        format!(
            "Processed {} frames in {} seconds, FPS {} fps",
            delta_frames,
            delta_time,
            Self::fps(delta_frames, delta_time)
        )
    }
}
