use std::{
    collections::VecDeque,
    num::NonZeroU32,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use hashbrown::HashMap;
use log::{debug, error, info};
use retina::NtpTimestamp;
use savant_core::primitives::{
    rust::{AttributeValue, VideoFrameProxy},
    Attribute, WithAttributes,
};

use crate::utils::{ts2epoch_duration, ONE_NS};

pub struct NtpSync {
    pub group_name: String,
    pub sync_window: VecDeque<(Duration, VideoFrameProxy, Vec<u8>)>,
    pub sync_window_duration: Duration,
    pub batch_duration: Duration,
    pub rtp_marks: HashMap<String, VecDeque<(i64, Duration, NonZeroU32)>>,
    pub batch_id: u64,
    pub skew_millis: HashMap<String, i64>,
    pub network_skew_correction: bool,
}

impl NtpSync {
    pub fn prune(&mut self, source_id: &str) {
        self.rtp_marks.remove(source_id);
        self.skew_millis.remove(source_id);
    }

    pub fn new(
        group_name: String,
        duration: Duration,
        batch_duration: Duration,
        network_skew_correction: bool,
    ) -> Self {
        Self {
            group_name,
            network_skew_correction,
            sync_window: VecDeque::new(),
            sync_window_duration: duration,
            batch_duration,
            rtp_marks: HashMap::new(),
            skew_millis: HashMap::new(),
            batch_id: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        }
    }

    pub fn add_rtp_mark(
        &mut self,
        source_id: &String,
        rtp_time: i64,
        ntp_time: NtpTimestamp,
        clock_rate: NonZeroU32,
    ) {
        let cam_epoch_duration = ts2epoch_duration(ntp_time, 0);
        let my_epoch_duration = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
        let diff = cam_epoch_duration
            .as_millis()
            .abs_diff(my_epoch_duration.as_millis()) as i64;
        let skew_millis = if my_epoch_duration >= cam_epoch_duration {
            diff
        } else {
            -diff
        };
        let current_skew = self
            .skew_millis
            .entry(source_id.clone())
            .or_insert_with(|| skew_millis);

        *current_skew = (*current_skew + skew_millis) / 2;

        info!(
            "Source_id: {}, NTP time: {:?}, my time: {:?}, diff (ours - cam): {:?} ms",
            source_id, cam_epoch_duration, my_epoch_duration, skew_millis
        );
        self.rtp_marks
            .entry(source_id.clone())
            .or_insert_with(VecDeque::new)
            .push_back((
                rtp_time,
                ts2epoch_duration(
                    ntp_time,
                    if self.network_skew_correction {
                        *current_skew
                    } else {
                        0
                    },
                ),
                clock_rate,
            ));
    }

    pub fn is_ready(&self, source_id: &String) -> bool {
        self.rtp_marks.contains_key(source_id)
    }

    pub fn prune_rtp_marks(&mut self, source_id: &String, rtp_time: i64) {
        let marks = self
            .rtp_marks
            .entry(source_id.clone())
            .or_insert_with(VecDeque::new);

        if marks.len() <= 1 {
            return;
        }

        let mut drain_history = 0;
        for (rtp, _, _) in marks.iter() {
            if *rtp <= rtp_time {
                drain_history += 1;
            } else {
                break;
            }
        }

        let drain_history = drain_history.min(marks.len() - 1);
        if drain_history > 1 {
            for _ in 0..drain_history - 1 {
                let elt = marks.pop_front().unwrap();
                info!(
                    "Pruned RTP mark {} for source_id={}, current RTP packet timestamp: {}",
                    elt.0, source_id, rtp_time
                );
            }
            info!(
                "Current RTP mark for source_id={} is {}, packet RTP is {}, Packet RTP - RTP mark = {}",
                source_id,
                marks.front().unwrap().0,
                rtp_time,
                rtp_time - marks.front().unwrap().0
            );
        }
    }

    pub fn current_rtp_mark(&self, source_id: &String) -> Option<(i64, Duration, NonZeroU32)> {
        let marks = self.rtp_marks.get(source_id);
        marks
            .map(|marks| {
                if marks.is_empty() {
                    return None;
                }
                let (rtp, ntp, clock_rate) = marks.front().unwrap();
                Some((*rtp, *ntp, *clock_rate))
            })
            .flatten()
    }

    pub fn add_frame(&mut self, rtp: i64, mut frame: VideoFrameProxy, frame_data: &[u8]) {
        let current_rtp_mark = self.current_rtp_mark(&frame.get_source_id());
        if let Some((rtp_mark, ntp_mark, clock_rate)) = current_rtp_mark {
            let diff = rtp - rtp_mark;
            if diff < 0 {
                error!(
                    "RTP time {} is less than RTP mark {}, diff: {}",
                    rtp, rtp_mark, diff
                );
                return;
            }
            let diff = (diff as f64 / clock_rate.get() as f64 * ONE_NS) as u64;
            let diff = Duration::from_nanos(diff);

            let ts = diff + ntp_mark;
            debug!(
                "Adding frame with RTP time {:0>16} and computed NTP timestamp {:?}, source_id={}",
                rtp,
                ts,
                frame.get_source_id()
            );
            frame.set_attribute(Attribute::new(
                "retina-rtsp",
                "ntp-timestamp-ns",
                vec![AttributeValue::string(&ts.as_nanos().to_string(), None)],
                &None,
                true,
                false,
            ));
            self.sync_window.push_back((ts, frame, frame_data.to_vec()));
            self.sync_window
                .make_contiguous()
                .sort_by(|(ts0, _, _), (ts1, _, _)| ts0.cmp(ts1));
        }
    }

    pub fn pull_frames(&mut self) -> Vec<(VideoFrameProxy, Duration, Vec<u8>)> {
        let mut res = HashMap::new();
        if self.sync_window.is_empty() {
            return res.into_values().collect();
        }
        let first = self.sync_window.front().unwrap().0;
        let last = self.sync_window.back().unwrap().0;
        let diff = last - first;
        debug!(
            "Pulling frames, current sync window duration: {:?}, required duration: {:?}, length: {}",
            diff, self.sync_window_duration, self.sync_window.len()
        );
        if diff > self.sync_window_duration {
            // we have window built and full
            // now we must read a batch of frames
            let mut current_duration = first;
            let batch_id = self.batch_id;
            let before_sync_window_len = self.sync_window.len();
            while current_duration - first < self.batch_duration {
                {
                    let (_, frame, _) = self.sync_window.front_mut().unwrap();
                    // no duplicates allowed
                    if res.contains_key(&frame.get_source_id()) {
                        break;
                    }
                }
                let (ts, frame, frame_data) = self.sync_window.pop_front().unwrap();
                res.insert(frame.get_source_id(), (frame, ts, frame_data));
                current_duration = ts;
            }
            let batch_sources = res.keys().cloned().collect::<Vec<_>>();
            let batch_participants = res
                .into_values()
                .map(|(mut frame, duration, frame_data)| {
                    frame.set_attribute(Attribute::new(
                        "retina-rtsp",
                        "batch-id",
                        vec![AttributeValue::string(&batch_id.to_string(), None)],
                        &None,
                        true,
                        false,
                    ));
                    frame.set_attribute(Attribute::new(
                        "retina-rtsp",
                        "batch-group-name",
                        vec![AttributeValue::string(&self.group_name, None)],
                        &None,
                        true,
                        false,
                    ));
                    frame.set_attribute(Attribute::new(
                        "retina-rtsp",
                        "batch-sources",
                        vec![AttributeValue::string_vector(
                            batch_sources
                                .as_slice()
                                .iter()
                                .map(|s| s.to_string())
                                .collect(),
                            None,
                        )],
                        &None,
                        true,
                        false,
                    ));
                    (frame, duration, frame_data)
                })
                .collect();
            self.batch_id += 1;
            let after_sync_window_len = self.sync_window.len();
            debug!(
                "Pulled {} frames from sync window, before len: {}, after len: {}",
                before_sync_window_len - after_sync_window_len,
                before_sync_window_len,
                after_sync_window_len
            );
            batch_participants
        } else {
            res.into_values().collect()
        }
    }
}
