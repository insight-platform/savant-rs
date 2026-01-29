use crate::best_ts;
use anyhow::{bail, Result};
use log::debug;
use savant_core::message::Message;
use serde::{Deserialize, Serialize};
use std::time::Instant;
use uuid::Uuid;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum JobStopCondition {
    #[serde(rename = "last_frame")]
    LastFrame {
        uuid: String,
        #[serde(skip)]
        uuid_u128: Option<u128>,
    },
    #[serde(rename = "frame_count")]
    FrameCount(usize),
    #[serde(rename = "key_frame_count")]
    KeyFrameCount(usize),
    #[serde(rename = "ts_delta_sec")]
    TSDeltaSec {
        max_delta_sec: f64,
        #[serde(skip)]
        first_pts: Option<i64>,
    },
    #[serde(rename = "real_time_delta_ms")]
    RealTimeDelta {
        configured_delta_ms: u64,
        #[serde(skip)]
        initial_ts: Option<Instant>,
    },
    #[serde(rename = "now")]
    Now,
    #[serde(rename = "never")]
    Never,
}

impl JobStopCondition {
    pub fn last_frame(uuid: u128) -> Self {
        JobStopCondition::LastFrame {
            uuid: Uuid::from_u128(uuid).to_string(),
            uuid_u128: None,
        }
    }

    pub fn frame_count(count: usize) -> Self {
        JobStopCondition::FrameCount(count)
    }

    pub fn key_frame_count(count: usize) -> Self {
        JobStopCondition::KeyFrameCount(count)
    }

    pub fn pts_delta_sec(max_delta: f64) -> Self {
        JobStopCondition::TSDeltaSec {
            max_delta_sec: max_delta,
            first_pts: None,
        }
    }

    pub fn real_time_delta_ms(configured_delta: u64) -> Self {
        JobStopCondition::RealTimeDelta {
            configured_delta_ms: configured_delta,
            initial_ts: None,
        }
    }

    pub fn setup(&mut self) -> Result<()> {
        match self {
            JobStopCondition::LastFrame { uuid, uuid_u128 } => {
                debug!("Setting up last frame stop condition with UUID: {}", uuid);
                *uuid_u128 = Some(Uuid::parse_str(uuid)?.as_u128());
            }
            JobStopCondition::RealTimeDelta { initial_ts, .. } => {
                let now = Instant::now();
                debug!(
                    "Setting up real time delta stop condition with initial timestamp: {:?}",
                    now
                );
                *initial_ts = Some(now);
            }
            _ => {}
        }
        Ok(())
    }

    pub fn check(&mut self, message: &Message) -> Result<bool> {
        if !message.is_video_frame() {
            return Ok(false);
        }
        let message = message.as_video_frame().unwrap();
        match self {
            JobStopCondition::LastFrame { uuid_u128, .. } => {
                if uuid_u128.is_none() {
                    bail!("UUID not set for last frame stop condition. Invoke setup() first.");
                }
                Ok(message.get_uuid_u128() >= uuid_u128.unwrap())
            }
            JobStopCondition::FrameCount(fc) => {
                if *fc == 1 {
                    return Ok(true);
                }
                *fc -= 1;
                Ok(false)
            }
            JobStopCondition::KeyFrameCount(kfc) => {
                if let Some(true) = message.get_keyframe() {
                    if *kfc == 1 {
                        return Ok(true);
                    }
                    *kfc -= 1;
                }
                Ok(false)
            }
            JobStopCondition::TSDeltaSec {
                max_delta_sec,
                first_pts,
            } => {
                if first_pts.is_none() {
                    *first_pts = Some(best_ts(&message));
                    return Ok(false);
                }
                let pts = best_ts(&message);
                let prev_pts = first_pts.unwrap();
                let pts_delta = pts.saturating_sub(prev_pts);
                let (time_base_n, time_base_d) = message.get_time_base();
                let pts_delta = pts_delta as f64 * time_base_n as f64 / time_base_d as f64;
                if pts_delta > *max_delta_sec {
                    return Ok(true);
                }
                Ok(false)
            }
            JobStopCondition::RealTimeDelta {
                configured_delta_ms,
                initial_ts,
            } => {
                if initial_ts.is_none() {
                    bail!("Initial timestamp not set for real time delta stop condition. Invoke setup() first.");
                }
                let elapsed = initial_ts.map(|i| i.elapsed().as_millis()).unwrap();
                if elapsed > *configured_delta_ms as u128 {
                    debug!(
                        "Elapsed time: {} ms, configured delta: {} ms",
                        elapsed, configured_delta_ms
                    );
                    return Ok(true);
                }
                Ok(false)
            }
            JobStopCondition::Now => Ok(true),
            JobStopCondition::Never => Ok(false),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::JobStopCondition;
    use crate::store::gen_properly_filled_frame;
    use anyhow::Result;
    use savant_core::utils::uuid_v7::incremental_uuid_v7;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_last_frame_stop_condition() -> Result<()> {
        let frame_before = gen_properly_filled_frame(true)?;
        thread::sleep(Duration::from_millis(1));
        let mut stop_condition = JobStopCondition::last_frame(incremental_uuid_v7().as_u128());
        stop_condition.setup()?;
        assert!(!stop_condition.check(&frame_before.to_message())?);
        thread::sleep(Duration::from_millis(1));
        let frame_after = gen_properly_filled_frame(true)?;
        assert!(stop_condition.check(&frame_after.to_message())?);
        Ok(())
    }

    #[test]
    fn test_frame_count_stop_condition() -> Result<()> {
        let frame = gen_properly_filled_frame(true)?;
        let mut stop_condition = JobStopCondition::frame_count(2);
        assert!(!stop_condition.check(&frame.to_message())?);
        assert!(stop_condition.check(&frame.to_message())?);
        Ok(())
    }

    #[test]
    fn test_key_frame_count_stop_condition() -> Result<()> {
        let mut frame = gen_properly_filled_frame(true)?;
        frame.set_keyframe(Some(true));
        let mut stop_condition = JobStopCondition::key_frame_count(2);
        assert!(!stop_condition.check(&frame.to_message())?);
        frame.set_keyframe(Some(false));
        assert!(!stop_condition.check(&frame.to_message())?);
        let key_frame = gen_properly_filled_frame(true)?;
        assert!(stop_condition.check(&key_frame.to_message())?);
        Ok(())
    }

    #[test]
    fn test_pts_delta_stop_condition() -> Result<()> {
        let mut frame = gen_properly_filled_frame(true)?;
        frame.set_time_base((1, 1_000_000))?;
        frame.set_pts(1_000_000)?;
        let mut stop_condition = JobStopCondition::pts_delta_sec(1.0);
        assert!(!stop_condition.check(&frame.to_message())?);
        frame.set_pts(1_700_000)?;
        assert!(!stop_condition.check(&frame.to_message())?);
        frame.set_pts(2_100_000)?;
        assert!(stop_condition.check(&frame.to_message())?);
        Ok(())
    }

    #[test]
    fn test_real_time_delta_stop_condition() -> Result<()> {
        let frame = gen_properly_filled_frame(true)?;
        let mut stop_condition = JobStopCondition::real_time_delta_ms(500);
        stop_condition.setup()?;
        assert!(!stop_condition.check(&frame.to_message())?);
        thread::sleep(Duration::from_millis(600));
        assert!(stop_condition.check(&frame.to_message())?);
        Ok(())
    }

    #[test]
    fn dump_all_stop_conditions() {
        let stop_conditions = vec![
            JobStopCondition::LastFrame {
                uuid: "uuid".to_string(),
                uuid_u128: None,
            },
            JobStopCondition::FrameCount(1),
            JobStopCondition::KeyFrameCount(1),
            JobStopCondition::TSDeltaSec {
                max_delta_sec: 1.0,
                first_pts: None,
            },
            JobStopCondition::RealTimeDelta {
                configured_delta_ms: 1,
                initial_ts: None,
            },
            JobStopCondition::Now,
            JobStopCondition::Never,
        ];
        for stop_condition in stop_conditions {
            println!("{}", serde_json::to_string_pretty(&stop_condition).unwrap());
        }
    }
}
