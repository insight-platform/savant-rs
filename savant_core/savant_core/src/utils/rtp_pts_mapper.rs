use std::fmt;
use std::time::Duration;

use thiserror::Error;

const RTP_MODULUS: i128 = 1i128 << 32;
const RTP_WRAP_THRESHOLD: u64 = 1u64 << 31;
const NANOS_PER_SECOND: i128 = 1_000_000_000;

/// Helper describing how many seconds one tick represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Timebase {
    pub numerator: i64,
    pub denominator: i64,
}

impl Timebase {
    pub const fn new_unchecked(numerator: i64, denominator: i64) -> Self {
        Self {
            numerator,
            denominator,
        }
    }

    pub fn new(numerator: i64, denominator: i64) -> Result<Self, RtpPtsMapperError> {
        if numerator <= 0 || denominator <= 0 {
            return Err(RtpPtsMapperError::InvalidTimebase {
                numerator,
                denominator,
            });
        }
        Ok(Self {
            numerator,
            denominator,
        })
    }

    fn as_i128(&self) -> (i128, i128) {
        (self.numerator as i128, self.denominator as i128)
    }
}

#[derive(Debug, Clone)]
pub struct RtpPtsMapper {
    rtp_timebase: Timebase,
    pts_timebase: Timebase,
    unwrapper: RtpUnwrapper,
    seed: Option<Seed>,
}

impl RtpPtsMapper {
    pub fn new(
        rtp_timebase: (i64, i64),
        pts_timebase: (i64, i64),
    ) -> Result<Self, RtpPtsMapperError> {
        Ok(Self {
            rtp_timebase: Timebase::new(rtp_timebase.0, rtp_timebase.1)?,
            pts_timebase: Timebase::new(pts_timebase.0, pts_timebase.1)?,
            unwrapper: RtpUnwrapper::default(),
            seed: None,
        })
    }

    pub fn with_seed(
        seed_rtp: u32,
        seed_ts: Duration,
        rtp_timebase: (i64, i64),
        pts_timebase: (i64, i64),
    ) -> Result<Self, RtpPtsMapperError> {
        let mut mapper = Self::new(rtp_timebase, pts_timebase)?;
        mapper.set_seed(seed_rtp, seed_ts)?;
        Ok(mapper)
    }

    pub fn set_seed(&mut self, rtp: u32, ts: Duration) -> Result<(), RtpPtsMapperError> {
        let unwrapped = self.unwrapper.unwrap(rtp)?;
        self.seed = Some(Seed {
            unwrapped_rtp: unwrapped,
            ts,
        });
        Ok(())
    }

    pub fn map(&mut self, rtp: u32) -> Result<RtpPtsMapping, RtpPtsMapperError> {
        let seed = self.seed.ok_or(RtpPtsMapperError::SeedMissing)?;
        let unwrapped = self.unwrapper.unwrap(rtp)?;
        let delta = unwrapped - seed.unwrapped_rtp;
        if delta < 0 {
            return Err(RtpPtsMapperError::NegativeDelta { delta });
        }
        let pts = self.delta_to_pts(delta)?;
        let ts_delta = self.delta_to_duration(delta)?;
        let ts = seed
            .ts
            .checked_add(ts_delta)
            .ok_or(RtpPtsMapperError::TimestampOverflow)?;
        Ok(RtpPtsMapping { pts, ts })
    }

    fn delta_to_pts(&self, delta: i128) -> Result<i64, RtpPtsMapperError> {
        let (rtp_num, rtp_den) = self.rtp_timebase.as_i128();
        let (pts_num, pts_den) = self.pts_timebase.as_i128();
        let numerator = delta
            .checked_mul(rtp_num)
            .and_then(|v| v.checked_mul(pts_den))
            .ok_or(RtpPtsMapperError::Overflow)?;
        let denominator = rtp_den
            .checked_mul(pts_num)
            .ok_or(RtpPtsMapperError::Overflow)?;
        let pts = (numerator + denominator / 2) / denominator;
        i64::try_from(pts).map_err(|_| RtpPtsMapperError::Overflow)
    }

    fn delta_to_duration(&self, delta: i128) -> Result<Duration, RtpPtsMapperError> {
        let (rtp_num, rtp_den) = self.rtp_timebase.as_i128();
        let numerator = delta
            .checked_mul(rtp_num)
            .and_then(|v| v.checked_mul(NANOS_PER_SECOND))
            .ok_or(RtpPtsMapperError::Overflow)?;
        let nanos = (numerator + rtp_den / 2) / rtp_den;
        if nanos < 0 {
            return Err(RtpPtsMapperError::NegativeDelta { delta: nanos });
        }
        let seconds = nanos / NANOS_PER_SECOND;
        let nanos_part = (nanos % NANOS_PER_SECOND) as u32;
        let seconds = u64::try_from(seconds).map_err(|_| RtpPtsMapperError::Overflow)?;
        Ok(Duration::new(seconds, nanos_part))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RtpPtsMapping {
    pub pts: i64,
    pub ts: Duration,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum RtpPtsMapperError {
    #[error("invalid timebase {numerator}/{denominator}, both parts must be positive")]
    InvalidTimebase { numerator: i64, denominator: i64 },
    #[error("mapper seed is not set")]
    SeedMissing,
    #[error("rtp delta is negative: {delta}")]
    NegativeDelta { delta: i128 },
    #[error("integer overflow while converting timestamps")]
    Overflow,
    #[error("resulting timestamp does not fit into Duration")]
    TimestampOverflow,
}

impl fmt::Display for Timebase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.numerator, self.denominator)
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct RtpUnwrapper {
    last_raw: Option<u32>,
    wraps: i128,
}

impl RtpUnwrapper {
    fn unwrap(&mut self, value: u32) -> Result<i128, RtpPtsMapperError> {
        if let Some(last) = self.last_raw {
            if value < last {
                let diff = (last - value) as u64;
                if diff > RTP_WRAP_THRESHOLD {
                    self.wraps = self
                        .wraps
                        .checked_add(1)
                        .ok_or(RtpPtsMapperError::Overflow)?;
                }
            }
        }
        self.last_raw = Some(value);
        Ok(self.wraps * RTP_MODULUS + value as i128)
    }
}

#[derive(Debug, Clone, Copy)]
struct Seed {
    unwrapped_rtp: i128,
    ts: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_basic_pts_and_ts() {
        let mut mapper =
            RtpPtsMapper::with_seed(0, Duration::from_secs(0), (1, 90_000), (1, 1_000_000))
                .unwrap();
        let mapping = mapper.map(90_000).unwrap();
        assert_eq!(mapping.pts, 1_000_000);
        assert_eq!(mapping.ts, Duration::from_secs(1));
    }

    #[test]
    fn handles_wraparound() {
        let mut mapper = RtpPtsMapper::with_seed(
            u32::MAX - 10,
            Duration::from_secs(0),
            (1, 90_000),
            (1, 1_000_000),
        )
        .unwrap();
        let mapping = mapper.map(20).unwrap();
        let expected_delta = 31; // 11 ticks to wrap + 20 ticks after wrap
        assert_eq!(mapping.pts, expected_delta * 1_000_000 / 90_000);
        let expected_ts =
            Duration::from_nanos((expected_delta as f64 / 90_000f64 * 1e9).round() as u64);
        assert_eq!(mapping.ts, expected_ts);
    }

    #[test]
    fn updates_seed() {
        let mut mapper =
            RtpPtsMapper::with_seed(0, Duration::from_secs(0), (1, 90_000), (1, 1_000_000))
                .unwrap();
        mapper.map(90_000).unwrap();
        mapper.set_seed(90_000, Duration::from_secs(5)).unwrap();
        let mapping = mapper.map(180_000).unwrap();
        assert_eq!(mapping.pts, 1_000_000);
        assert_eq!(mapping.ts, Duration::from_secs(6));
    }
}
