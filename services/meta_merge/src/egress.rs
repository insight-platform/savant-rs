use crate::configuration::ServiceConfiguration;

mod merge_queue;
mod payload;

// mod merge_queue;

// pub struct Egress {
//     queue: merge_queue::MergeQueue,
// }

// impl Egress {
//     pub fn new(config: &ServiceConfiguration) -> Self {
//         Self {
//             queue: merge_queue::MergeQueue::new(config.common.queue.max_duration),
//         }
//     }
// }
