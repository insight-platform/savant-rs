use std::collections::VecDeque;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use hashbrown::HashMap;
use log::info;
use parking_lot::{Mutex, MutexGuard};

use crate::rwlock::SavantRwLock;

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
pub struct StageProcessingStat {
    pub stage_name: String,
    pub queue_length: usize,
    pub frame_counter: usize,
    pub object_counter: usize,
    pub batch_counter: usize,
}

#[derive(Debug, Clone, Default)]
pub struct StageLatencyStat {
    pub stage_name: String,
    pub latencies: HashMap<usize, StageLatencyMeasurements>,
}

#[derive(Debug, Clone, Default)]
pub struct StageLatencyMeasurements {
    pub source_stage_name: Option<String>,
    pub min_latency: Duration,
    pub max_latency: Duration,
    pub accumulated_latency: Duration,
    pub count: usize,
}

impl StageLatencyStat {
    pub fn new(name: String) -> Self {
        Self {
            stage_name: name,
            ..Default::default()
        }
    }

    pub fn record_latency(&mut self, source_stage: usize, latency: Duration) {
        let measurements =
            self.latencies
                .entry(source_stage)
                .or_insert_with(|| StageLatencyMeasurements {
                    source_stage_name: None,
                    min_latency: latency,
                    max_latency: latency,
                    accumulated_latency: Duration::from_secs(0),
                    count: 0,
                });
        measurements.min_latency = std::cmp::min(measurements.min_latency, latency);
        measurements.max_latency = std::cmp::max(measurements.max_latency, latency);
        measurements.accumulated_latency += latency;
        measurements.count += 1;
    }

    pub fn log_latencies(&self) {
        let none = "None".to_string();
        for (_, latency) in &self.latencies {
            info!(
                "⏱ {:<32} > {:<32} | min {:>8} micros, max {:>8} micros, avg {:>8} micros,    count {:>8}",
                latency.source_stage_name.as_ref().unwrap_or(&none),
                self.stage_name,
                latency.min_latency.as_micros(),
                latency.max_latency.as_micros(),
                latency.accumulated_latency.as_micros() / latency.count as u128,
                latency.count,
            );
        }
    }
}

impl StageProcessingStat {
    pub fn new(name: String) -> Self {
        StageProcessingStat {
            stage_name: name,
            ..Default::default()
        }
    }

    pub fn log_stats(&self) {
        info!(
            "📊 {:<32} > queue {:>8}, frames {:>8}, objects {:>8}, batches {:>8}",
            self.stage_name,
            self.queue_length,
            self.frame_counter,
            self.object_counter,
            self.batch_counter,
        );
    }
}

#[derive(Debug, Clone)]
pub struct FrameProcessingStatRecord {
    pub id: i64,
    pub record_type: FrameProcessingStatRecordType,
    pub ts: i64,
    pub frame_no: usize,
    pub object_counter: usize,
    pub stage_stats: Vec<(StageProcessingStat, StageLatencyStat)>,
}

impl FrameProcessingStatRecord {
    pub(crate) fn log_stage_stats(&self) {
        for (sps, _) in &self.stage_stats {
            sps.log_stats();
        }
        for (_, sls) in &self.stage_stats {
            sls.log_latencies();
        }
    }
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

    pub fn get_records<F>(&self, max_n: usize, filter: F) -> Vec<FrameProcessingStatRecord>
    where
        F: FnMut(&&FrameProcessingStatRecord) -> bool,
    {
        self.processing_history
            .iter()
            .filter(filter)
            .take(max_n)
            .cloned()
            .collect()
    }

    pub fn get_max_length(&self) -> usize {
        self.max_length
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

    pub fn register_frame(
        &mut self,
        object_counter: usize,
        last: bool,
    ) -> Option<FrameProcessingStatRecord> {
        if self.is_active() {
            self.current_frame += 1;
            self.object_counter += object_counter;
        }

        match (self.frame_period, self.last_frame) {
            (Some(frame_period), Some(last_frame)) => {
                let frame_no = self.current_frame;
                if frame_no - last_frame >= frame_period || last {
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

    pub fn register_ts(&mut self, last: bool) -> Option<FrameProcessingStatRecord> {
        match (self.timestamp_period, self.last_ts) {
            (Some(timestamp_period), Some(last_ts)) => {
                let ts = self.time_counter.get_current_time();
                let frame_no = self.current_frame;
                if ts - last_ts >= timestamp_period || last {
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

pub type StageStats = Arc<SavantRwLock<(StageProcessingStat, StageLatencyStat)>>;

#[derive(Debug)]
pub struct Stats {
    collector: Arc<Mutex<StatsCollector>>,
    generator: Arc<Mutex<StatsGenerator>>,
    time_thread: Option<std::thread::JoinHandle<()>>,
    shutdown: Arc<OnceLock<()>>,
    stage_stats: Arc<Mutex<Vec<StageStats>>>,
}

impl Default for Stats {
    fn default() -> Self {
        Stats::new(100, Some(1000), Some(1000))
    }
}

fn log_ts_fps(collector: &mut MutexGuard<StatsCollector>) {
    let last_records = collector.get_records(2, |r| {
        matches!(
            r.record_type,
            FrameProcessingStatRecordType::Timestamp | FrameProcessingStatRecordType::Initial
        )
    });
    if last_records.len() == 2 {
        let time_delta = (last_records[0].ts - last_records[1].ts) as f64 / 1000.0;
        let frame_delta = last_records[0].frame_no - last_records[1].frame_no;
        let object_delta = last_records[0].object_counter - last_records[1].object_counter;
        info!(
            "📊 Time-based FPS counter triggered: FPS = {:.2}, OPS = {:.2}, frame_delta = {}, time_delta = {} sec , period=[{}, {}] ms",
            frame_delta as f64 / time_delta,
            object_delta as f64 / time_delta,
            frame_delta,
            time_delta,
            last_records[1].ts,
            last_records[0].ts
        );
    }
}

fn log_frame_fps(collector: &mut MutexGuard<StatsCollector>) {
    let last_records = collector.get_records(2, |r| {
        matches!(
            r.record_type,
            FrameProcessingStatRecordType::Frame | FrameProcessingStatRecordType::Initial
        )
    });
    if last_records.len() == 2 {
        let time_delta = (last_records[0].ts - last_records[1].ts) as f64 / 1000.0;
        let frame_delta = last_records[0].frame_no - last_records[1].frame_no;
        let object_delta = last_records[0].object_counter - last_records[1].object_counter;
        info!(
            "📊 Frame-based FPS counter triggered: FPS = {:.2}, OPS = {:.2}, frame_delta = {}, time_delta = {} sec, period=[{}, {}] ms",
            frame_delta as f64 / time_delta,
            object_delta as f64 / time_delta,
            frame_delta,
            time_delta,
            last_records[1].ts,
            last_records[0].ts
        );
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
            let res = thread_generator.lock().register_ts(false);
            if let Some(mut r) = res {
                r.stage_stats = Stats::collect_stage_stats(&thread_stage_stats);
                r.log_stage_stats();
                let mut thread_collector_bind = thread_collector.lock();
                thread_collector_bind.add_record(r);
                log_ts_fps(&mut thread_collector_bind);
            }
            std::thread::sleep(Duration::from_millis(1));
        }));

        Stats {
            collector,
            generator,
            time_thread,
            shutdown,
            stage_stats,
        }
    }

    pub fn add_stage_stats(
        &self,
        stat: Arc<SavantRwLock<(StageProcessingStat, StageLatencyStat)>>,
    ) {
        self.stage_stats.lock().push(stat);
    }

    pub fn collect_stage_stats(
        stats: &Arc<Mutex<Vec<StageStats>>>,
    ) -> Vec<(StageProcessingStat, StageLatencyStat)> {
        stats.lock().iter().map(|s| s.read().clone()).collect()
    }

    pub fn kick_off(&self) {
        let res = self.generator.lock().kick_off();
        if let Some(r) = res {
            self.collector.lock().add_record(r);
        }
    }

    pub fn register_frame(&self, object_counter: usize) {
        let res = self.generator.lock().register_frame(object_counter, false);
        if let Some(mut r) = res {
            r.stage_stats = Stats::collect_stage_stats(&self.stage_stats);
            r.log_stage_stats();
            let mut collector_bind = self.collector.lock();
            collector_bind.add_record(r);
            log_frame_fps(&mut collector_bind);
        }
    }

    pub fn get_records(&self, max_n: usize) -> Vec<FrameProcessingStatRecord> {
        self.collector.lock().get_records(max_n, |_| true)
    }

    pub fn get_records_newer_than(&self, id: i64) -> Vec<FrameProcessingStatRecord> {
        let bind = self.collector.lock();
        let history_length = bind.get_max_length();
        bind.get_records(history_length, |r| r.id > id)
    }

    pub fn log_final_fps(&self) {
        // add final record for frames if they are configured
        let mut generator_bind = self.generator.lock();
        if generator_bind.frame_period.is_some() {
            let res = generator_bind.register_frame(0, true);
            if let Some(mut r) = res {
                r.stage_stats = Stats::collect_stage_stats(&self.stage_stats);
                r.log_stage_stats();
                let mut collector_bind = self.collector.lock();
                collector_bind.add_record(r);
                log_frame_fps(&mut collector_bind);
            }
        }

        if generator_bind.timestamp_period.is_some() {
            let res = generator_bind.register_ts(true);
            if let Some(mut r) = res {
                r.stage_stats = Stats::collect_stage_stats(&self.stage_stats);
                let mut collector_bind = self.collector.lock();
                collector_bind.add_record(r);
                log_ts_fps(&mut collector_bind);
            }
        }
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
        let records = stats_collector.get_records(20, |_| true);
        assert_eq!(records.len(), 10);
        for i in 0..10 {
            assert_eq!(records[i].ts, 19 - i as i64);
            assert_eq!(records[i].frame_no, 19 - i);
        }
    }

    #[test]
    fn test_frame_based_stats_generator() {
        let mut generator = StatsGenerator::new(Some(5), None);
        let frame_rec = generator.register_frame(10, false);
        assert!(frame_rec.is_none(), "Before kick off nothings happens");
        let ts_rec = generator.register_ts(false);
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
            let frame_rec = generator.register_frame(5, false);
            assert!(frame_rec.is_none(), "Not enough frames ingested");
        }
        let frame_rec = generator.register_frame(5, false);
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
            .flat_map(|_| generator.register_frame(1, false))
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
        let frame_rec = generator.register_frame(0, false);
        assert!(frame_rec.is_none(), "Before kick off nothings happens");
        let ts_rec = generator.register_ts(false);
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
            let ts_rec = generator.register_ts(false);
            assert!(ts_rec.is_none(), "Not enough timestamps ingested");
        }
        generator.register_frame(0, false);
        generator.time_counter.update_time(20);
        let ts_rec = generator.register_ts(false);
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
        generator.register_frame(0, false);
        generator.time_counter.update_time(40);
        let ts_rec = generator.register_ts(false);
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
