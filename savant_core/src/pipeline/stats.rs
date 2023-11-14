use std::collections::VecDeque;
use std::time::SystemTime;

pub struct FrameProcessingRecord {
    ts: i64,
    frame: i64,
}

pub struct StatsCollector {
    max_length: usize,
    processing_history: VecDeque<FrameProcessingRecord>,
}

impl StatsCollector {
    pub fn new(max_length: usize) -> Self {
        StatsCollector {
            max_length,
            processing_history: VecDeque::with_capacity(max_length),
        }
    }

    pub fn add_record(&mut self, ts: i64, frame: i64) {
        self.processing_history
            .push_front(FrameProcessingRecord { ts, frame });
    }

    pub fn get_records(&mut self) -> &VecDeque<FrameProcessingRecord> {
        self.processing_history.truncate(self.max_length);
        self.processing_history.shrink_to_fit();
        &self.processing_history
    }
}

#[derive(Default)]
pub struct StatsGenerator {
    frame_period: Option<i64>,
    timestamp_period: Option<i64>,
    last_ts: Option<i64>,
    last_frame: Option<i64>,
    current_frame: i64,
}

impl StatsGenerator {
    pub fn new(frame_period: Option<i64>, timestamp_period: Option<i64>) -> Self {
        StatsGenerator {
            frame_period,
            timestamp_period,
            ..Default::default()
        }
    }

    pub fn kick_off(&mut self) -> Option<FrameProcessingRecord> {
        if self.last_ts.is_some() {
            return None;
        }

        let ts = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;
        self.last_ts = Some(ts);
        self.last_frame = Some(0);
        self.current_frame = 0;
        Some(FrameProcessingRecord { ts, frame: 0 })
    }

    pub fn register_frame(&mut self) -> Option<FrameProcessingRecord> {
        match (self.frame_period, self.last_frame) {
            (Some(frame_period), Some(last_frame)) => {
                self.current_frame += 1;
                let frame_no = self.current_frame;

                if frame_no - last_frame >= frame_period {
                    let ts = SystemTime::now()
                        .duration_since(SystemTime::UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as i64;
                    self.last_frame = Some(frame_no);
                    Some(FrameProcessingRecord {
                        ts,
                        frame: frame_no,
                    })
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    pub fn register_ts(&mut self) -> Option<FrameProcessingRecord> {
        match (self.timestamp_period, self.last_ts) {
            (Some(timestamp_period), Some(last_ts)) => {
                let ts = SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as i64;
                let frame = self.last_frame.unwrap();
                if ts - last_ts >= timestamp_period {
                    self.last_ts = Some(ts);
                    Some(FrameProcessingRecord { ts, frame })
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stats_collector() {
        let mut stats_collector = StatsCollector::new(10);
        for i in 0..20 {
            stats_collector.add_record(i, i);
        }
        let records = stats_collector.get_records();
        assert_eq!(records.len(), 10);
        for i in 0..10 {
            assert_eq!(records[i].ts, 19 - i as i64);
            assert_eq!(records[i].frame, 19 - i as i64);
        }
    }

    #[test]
    fn test_frame_based_stats_generator() {
        let mut generator = StatsGenerator::new(Some(5), None);
        let frame_rec = generator.register_frame();
        assert!(frame_rec.is_none());
        let ts_rec = generator.register_ts();
        assert!(ts_rec.is_none());
    }
}
