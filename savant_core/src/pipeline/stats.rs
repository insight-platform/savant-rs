use log::info;
use parking_lot::{Mutex, RwLock};
use std::collections::VecDeque;
use std::sync::{Arc, OnceLock};

#[cfg(test)]
#[derive(Default, Debug)]
struct TimeCounter {
    current_time: i64,
}

#[cfg(test)]
impl TimeCounter {
    pub fn update_time(&mut self, time: i64) {
        self.current_time = time;
    }

    pub fn get_current_time(&mut self) -> i64 {
        self.current_time
    }
}

#[cfg(not(test))]
#[derive(Default, Debug)]
struct TimeCounter;
#[cfg(not(test))]
impl TimeCounter {
    pub fn get_current_time(&self) -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum FrameProcessingStatRecordType {
    Initial,
    Frame,
    Timestamp,
}

#[derive(Debug, Clone, Default)]
pub struct StageStat {
    pub stage_name: String,
    pub queue_length: usize,
    pub frame_counter: usize,
    pub object_counter: usize,
    pub batch_counter: usize,
}

impl StageStat {
    pub fn new(name: String) -> Self {
        StageStat {
            stage_name: name,
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone)]
pub struct FrameProcessingStatRecord {
    pub id: i64,
    pub record_type: FrameProcessingStatRecordType,
    pub ts: i64,
    pub frame_no: usize,
    pub object_counter: usize,
    pub stage_stats: Vec<StageStat>,
}

#[derive(Debug)]
pub struct StatsCollector {
    max_length: usize,
    processing_history: VecDeque<FrameProcessingStatRecord>,
}

impl StatsCollector {
    pub fn new(max_length: usize) -> Self {
        StatsCollector {
            max_length,
            processing_history: VecDeque::with_capacity(max_length),
        }
    }

    pub fn add_record(&mut self, r: FrameProcessingStatRecord) {
        self.processing_history.push_front(r);
        if self.processing_history.len() > self.max_length {
            self.processing_history.pop_back();
        }
    }

    pub fn get_records(&mut self, max_n: usize) -> Vec<FrameProcessingStatRecord> {
        self.processing_history
            .iter()
            .take(max_n)
            .cloned()
            .collect()
    }
}

#[derive(Default, Debug)]
pub struct StatsGenerator {
    frame_period: Option<i64>,
    timestamp_period: Option<i64>,
    last_ts: Option<i64>,
    last_frame: Option<i64>,
    current_frame: i64,
    time_counter: TimeCounter,
    record_counter: i64,
    object_counter: usize,
}

impl StatsGenerator {
    pub fn new(frame_period: Option<i64>, timestamp_period: Option<i64>) -> Self {
        StatsGenerator {
            frame_period,
            timestamp_period,
            ..Default::default()
        }
    }

    #[inline]
    fn inc_record_counter(&mut self) -> i64 {
        let id = self.record_counter;
        self.record_counter += 1;
        id
    }

    pub fn kick_off(&mut self) -> Option<FrameProcessingStatRecord> {
        if self.last_ts.is_some() {
            return None;
        }

        let ts = self.time_counter.get_current_time();
        self.last_ts = Some(ts);
        self.last_frame = Some(0);
        self.current_frame = 0;

        Some(FrameProcessingStatRecord {
            id: self.inc_record_counter(),
            record_type: FrameProcessingStatRecordType::Initial,
            ts,
            frame_no: 0,
            object_counter: 0,
            stage_stats: Vec::new(),
        })
    }

    pub fn is_active(&self) -> bool {
        self.last_ts.is_some()
    }

    pub fn register_frame(&mut self, object_counter: usize) -> Option<FrameProcessingStatRecord> {
        if self.is_active() {
            self.current_frame += 1;
            self.object_counter += object_counter;
        }

        match (self.frame_period, self.last_frame) {
            (Some(frame_period), Some(last_frame)) => {
                let frame_no = self.current_frame;
                if frame_no - last_frame >= frame_period {
                    let ts = self.time_counter.get_current_time();
                    self.last_frame = Some(frame_no);
                    Some(FrameProcessingStatRecord {
                        id: self.inc_record_counter(),
                        record_type: FrameProcessingStatRecordType::Frame,
                        ts,
                        frame_no: frame_no as usize,
                        object_counter: self.object_counter,
                        stage_stats: Vec::new(),
                    })
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    pub fn register_ts(&mut self) -> Option<FrameProcessingStatRecord> {
        match (self.timestamp_period, self.last_ts) {
            (Some(timestamp_period), Some(last_ts)) => {
                let ts = self.time_counter.get_current_time();
                let frame_no = self.current_frame;
                if ts - last_ts >= timestamp_period {
                    self.last_ts = Some(ts);
                    Some(FrameProcessingStatRecord {
                        id: self.inc_record_counter(),
                        record_type: FrameProcessingStatRecordType::Timestamp,
                        ts,
                        frame_no: frame_no as usize,
                        object_counter: self.object_counter,
                        stage_stats: Vec::new(),
                    })
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

#[derive(Debug)]
pub struct Stats {
    collector: Arc<Mutex<StatsCollector>>,
    generator: Arc<Mutex<StatsGenerator>>,
    time_thread: Option<std::thread::JoinHandle<()>>,
    shutdown: Arc<OnceLock<()>>,
    stage_stats: Arc<Mutex<Vec<Arc<RwLock<StageStat>>>>>,
}

impl Default for Stats {
    fn default() -> Self {
        Stats::new(100, Some(1000), Some(1000))
    }
}

impl Stats {
    pub fn new(
        stats_history: usize,
        frame_period: Option<i64>,
        timestamp_period: Option<i64>,
    ) -> Self {
        let generator = Arc::new(parking_lot::Mutex::new(StatsGenerator::new(
            frame_period,
            timestamp_period,
        )));
        let collector = Arc::new(parking_lot::Mutex::new(StatsCollector::new(stats_history)));
        let shutdown = Arc::new(OnceLock::new());
        let stage_stats = Arc::new(Mutex::new(Vec::new()));

        let thread_generator = generator.clone();
        let thread_collector = collector.clone();
        let thread_shutdown = shutdown.clone();
        let thread_stage_stats = stage_stats.clone();

        let time_thread = Some(std::thread::spawn(move || loop {
            if thread_shutdown.get().is_some() {
                break;
            }
            let res = thread_generator.lock().register_ts();
            if let Some(mut r) = res {
                r.stage_stats = Stats::collect_stage_stats(&thread_stage_stats);
                thread_collector.lock().add_record(r);
                let last_records = thread_collector.lock().get_records(2);
                if last_records.len() == 2 {
                    let time_delta = last_records[0].ts - last_records[1].ts;
                    let frame_delta = last_records[0].frame_no - last_records[1].frame_no;
                    let object_delta =
                        last_records[0].object_counter - last_records[1].object_counter;
                    info!(
                        "Time-based FPS counter triggered: FPS = {}, OPS = {}, frame_delta = {}, time_delta = {}, period=[{}, {}]",
                        frame_delta as f64 / time_delta as f64,
                        object_delta as f64 / time_delta as f64,
                        frame_delta,
                        time_delta,
                        last_records[1].ts,
                        last_records[0].ts
                    );
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }));

        Stats {
            collector,
            generator,
            time_thread,
            shutdown,
            stage_stats,
        }
    }

    pub fn add_stage_stats(&self, stat: Arc<RwLock<StageStat>>) {
        self.stage_stats.lock().push(stat);
    }

    pub fn collect_stage_stats(stats: &Arc<Mutex<Vec<Arc<RwLock<StageStat>>>>>) -> Vec<StageStat> {
        stats.lock().iter().map(|s| s.read().clone()).collect()
    }

    pub fn kick_off(&self) {
        let res = self.generator.lock().kick_off();
        if let Some(r) = res {
            self.collector.lock().add_record(r);
        }
    }

    pub fn register_frame(&self, object_counter: usize) {
        let res = self.generator.lock().register_frame(object_counter);
        if let Some(mut r) = res {
            r.stage_stats = Stats::collect_stage_stats(&self.stage_stats);
            self.collector.lock().add_record(r);
            let last_records = self.collector.lock().get_records(2);
            if last_records.len() == 2 {
                let time_delta = last_records[0].ts - last_records[1].ts;
                let frame_delta = last_records[0].frame_no - last_records[1].frame_no;
                let object_delta = last_records[0].object_counter - last_records[1].object_counter;
                info!(
                    "Frame-based FPS counter triggered: FPS = {}, OPS = {}, frame_delta = {}, time_delta = {}, period=[{}, {}]",
                    frame_delta as f64 / time_delta as f64,
                    object_delta as f64 / time_delta as f64,
                    frame_delta,
                    time_delta,
                    last_records[0].ts,
                    last_records[1].ts
                );
            }
        }
    }

    pub fn get_records(&self, max_n: usize) -> Vec<FrameProcessingStatRecord> {
        self.collector.lock().get_records(max_n)
    }
}

impl Drop for Stats {
    fn drop(&mut self) {
        self.shutdown.get_or_init(|| ());
        let handle = self.time_thread.take().unwrap();
        handle.join().expect("Failed to join stats thread");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_sync_stats() {
        Stats::default();
    }

    #[test]
    fn test_stats_collector() {
        let mut stats_collector = StatsCollector::new(10);
        for i in 0..20 {
            stats_collector.add_record(FrameProcessingStatRecord {
                id: 0,
                record_type: FrameProcessingStatRecordType::Frame,
                ts: i,
                frame_no: i as usize,
                object_counter: 0,
                stage_stats: Vec::new(),
            });
        }
        let records = stats_collector.get_records(20);
        assert_eq!(records.len(), 10);
        for i in 0..10 {
            assert_eq!(records[i].ts, 19 - i as i64);
            assert_eq!(records[i].frame_no, 19 - i);
        }
    }

    #[test]
    fn test_frame_based_stats_generator() {
        let mut generator = StatsGenerator::new(Some(5), None);
        let frame_rec = generator.register_frame(10);
        assert!(frame_rec.is_none(), "Before kick off nothings happens");
        let ts_rec = generator.register_ts();
        assert!(ts_rec.is_none(), "Before kick off nothing happens");
        let frame_rec = generator.kick_off();
        assert!(
            frame_rec.is_some(),
            "Kick off happened, expected initial record"
        );
        assert_eq!(
            frame_rec.unwrap().record_type,
            FrameProcessingStatRecordType::Initial,
            "Expected initial record"
        );
        assert!(generator.is_active(), "Generator is active after kick off");

        let frame_rec = generator.kick_off();
        assert!(frame_rec.is_none());

        generator.time_counter.update_time(10);
        for _ in 0..4 {
            let frame_rec = generator.register_frame(5);
            assert!(frame_rec.is_none(), "Not enough frames ingested");
        }
        let frame_rec = generator.register_frame(5);
        assert!(frame_rec.is_some(), "Frame record expected");
        assert!(matches!(
            frame_rec,
            Some(FrameProcessingStatRecord {
                id,
                record_type,
                ts,
                frame_no,
                object_counter,
                stage_stats: _
            }) if record_type == FrameProcessingStatRecordType::Frame && ts == 10 && frame_no == 5 && id == 1 && object_counter == 25
        ));
        generator.time_counter.update_time(20);
        let mut frames = (0..5)
            .into_iter()
            .flat_map(|_| generator.register_frame(1))
            .collect::<Vec<_>>();
        assert_eq!(frames.len(), 1);
        let frame = frames.pop();
        assert!(matches!(
            frame,
            Some(FrameProcessingStatRecord {
                id,
                record_type,
                ts,
                frame_no,
                object_counter,
                stage_stats: _
            }) if record_type == FrameProcessingStatRecordType::Frame && ts == 20 && frame_no == 10 && id == 2 && object_counter == 30
        ));
    }

    #[test]
    fn test_timestamp_based_stats_generator() {
        let mut generator = StatsGenerator::new(None, Some(20));
        let frame_rec = generator.register_frame(0);
        assert!(frame_rec.is_none(), "Before kick off nothings happens");
        let ts_rec = generator.register_ts();
        assert!(ts_rec.is_none(), "Before kick off nothing happens");
        let frame_rec = generator.kick_off();
        assert!(
            frame_rec.is_some(),
            "Kick off happened, expected initial record"
        );
        assert_eq!(
            frame_rec.unwrap().record_type,
            FrameProcessingStatRecordType::Initial,
            "Expected initial record"
        );
        assert!(generator.is_active(), "Generator is active after kick off");

        let frame_rec = generator.kick_off();
        assert!(frame_rec.is_none());

        for ts in 16..20 {
            generator.time_counter.update_time(ts);
            let ts_rec = generator.register_ts();
            assert!(ts_rec.is_none(), "Not enough timestamps ingested");
        }
        generator.register_frame(0);
        generator.time_counter.update_time(20);
        let ts_rec = generator.register_ts();
        assert!(ts_rec.is_some(), "Timestamp record expected");
        assert!(matches!(
            ts_rec,
            Some(FrameProcessingStatRecord {
                id,
                record_type,
                ts,
                frame_no,
                object_counter: _,
                stage_stats: _
            }) if record_type == FrameProcessingStatRecordType::Timestamp && ts == 20 && frame_no == 1 && id == 1
        ));
        generator.register_frame(0);
        generator.time_counter.update_time(40);
        let ts_rec = generator.register_ts();
        assert!(ts_rec.is_some(), "Timestamp record expected");
        assert!(matches!(
            ts_rec,
            Some(FrameProcessingStatRecord {
                id,
                record_type,
                ts,
                frame_no,
                object_counter: _,
                stage_stats: _
            }) if record_type == FrameProcessingStatRecordType::Timestamp && ts == 40 && frame_no == 2 && id == 2
        ));
    }
}
