use hashbrown::HashMap;
use log::{debug, warn};
use savant_core::primitives::rust::VideoFrameProxy;
use std::{
    collections::VecDeque,
    time::{Duration, SystemTime},
};
pub struct Syncer {
    duration_queues: HashMap<String, VecDeque<(VideoFrameProxy, Vec<u8>)>>,
    sync_queues: HashMap<String, VecDeque<(VideoFrameProxy, Vec<u8>)>>,
    last_sent: HashMap<String, SystemTime>,
}

impl Syncer {
    pub fn prune(&mut self, source_id: &str) {
        self.duration_queues.remove(source_id);
        self.sync_queues.remove(source_id);
        self.last_sent.remove(source_id);
    }

    pub fn new() -> Self {
        Self {
            duration_queues: HashMap::new(),
            sync_queues: HashMap::new(),
            last_sent: HashMap::new(),
        }
    }

    pub fn add_frame(
        &mut self,
        frame: VideoFrameProxy,
        data: Vec<u8>,
    ) -> Option<(VideoFrameProxy, Vec<u8>)> {
        if let Some((frame, data)) = self.add_frame_to_duration_queue(frame, data) {
            self.add_frame_to_send_queue(frame, data)
        } else {
            None
        }
    }

    fn add_frame_to_send_queue(
        &mut self,
        frame: VideoFrameProxy,
        data: Vec<u8>,
    ) -> Option<(VideoFrameProxy, Vec<u8>)> {
        let source_id = frame.get_source_id();

        let sync_queue = self
            .sync_queues
            .entry(source_id.clone())
            .or_insert_with(VecDeque::new);
        sync_queue.push_back((frame, data));

        self.last_sent
            .entry(source_id.clone())
            .or_insert(SystemTime::now());

        let next_frame_duration =
            Duration::from_nanos(sync_queue.front().unwrap().0.get_duration().unwrap() as u64);

        let last_sent_time = self.last_sent[&source_id];
        let elapsed = last_sent_time.elapsed().unwrap();
        if elapsed >= next_frame_duration {
            let (frame, data) = sync_queue.pop_front().unwrap();
            self.last_sent
                .entry(source_id.clone())
                .or_insert(SystemTime::now());
            debug!(
                "Source ID: {}, Sending frame with PTS: {} to sink, queue size: {}",
                source_id,
                frame.get_pts(),
                sync_queue.len()
            );
            Some((frame, data))
        } else {
            None
        }
    }

    fn add_frame_to_duration_queue(
        &mut self,
        frame: VideoFrameProxy,
        data: Vec<u8>,
    ) -> Option<(VideoFrameProxy, Vec<u8>)> {
        let source_id = frame.get_source_id();
        let queue = self
            .duration_queues
            .entry(source_id.clone())
            .or_insert_with(VecDeque::new);
        queue.push_back((frame, data));

        if queue.len() == 2 {
            let (mut frame, data) = queue.pop_front().unwrap();
            let (second_frame, _) = queue.front().unwrap();
            let duration = second_frame.get_pts() - frame.get_pts();
            if duration <= 0 {
                warn!(
                    "Source ID: {}, PTS: {}, Duration is less than 0, this should never happen! Frame will be dropped!",
                    source_id, frame.get_pts()
                );
                return None;
            }

            debug!(
                "Source ID: {}, Setting duration for frame with PTS: {} to: {}, next frame PTS: {}",
                source_id,
                frame.get_pts(),
                duration,
                second_frame.get_pts()
            );
            frame.set_duration(Some(duration));
            Some((frame, data))
        } else if queue.len() > 2 {
            panic!("Queue size is greater than 2, this should never happen");
        } else {
            None
        }
    }
}
