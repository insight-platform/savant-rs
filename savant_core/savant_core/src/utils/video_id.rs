//! Composite 128-bit video frame identifier intended to replace
//! UUIDv7-based frame ids during the migration tracked alongside this
//! module.
//!
//! Encoded as a fully RFC 9562 §5.8 (UUIDv8) compliant UUID — the
//! version nibble at bits `[79:76]` is fixed at `0b1000` (= 8) and
//! the variant bits at `[63:62]` are fixed at `0b10`. The remaining
//! 122 application-defined bits hold our four fields:
//!
//! ```text
//!  127        96 95     80 79  76 75       64 63   62 61    48 47    40 39     0
//! +------------+---------+------+-----------+------+--------+--------+---------+
//! |   crc32    | ts_ms   | ver  |  ts_ms    | var  | ts_ms  | epoch  |   pts   |
//! |   32 bits  | hi: 16  | =1000| mid: 12   | =10  | lo: 14 |  8 b   |  40 b   |
//! +------------+---------+------+-----------+------+--------+--------+---------+
//! ```
//!
//! * `crc32` — `crc32(source_id)`. Same value for every frame of a
//!   given source.
//! * `ts_ms` — wall-clock ms of the GOP's keyframe at the demuxer.
//!   42 bits total (≈ 139 years from epoch — wraps in year 2109),
//!   stored across three slices that bracket the fixed
//!   version/variant nibbles. Constant for every frame in the same
//!   GOP. Strict-monotonic per source via the disorder-avoidance
//!   pattern used by `incremental_uuid_v7`.
//! * `epoch` — bumps when the demuxer detects a PTS reset for the
//!   source.
//! * `pts` — frame PTS, low-order tail. 40 bits is enough for any
//!   single GOP at common timebases.
//!
//! The version/variant bits are constant, so the u128 byte order
//! still sorts as `(crc32, ts_ms, epoch, pts)` — the goals from the
//! pre-v8 layout (within-GOP PTS-sort, cross-GOP keyframe-arrival
//! sort, reset tolerance via wall-clock dominance) are preserved.

use std::num::NonZeroUsize;

use crc32fast::Hasher as Crc32;
use lazy_static::lazy_static;
use lru::LruCache;
use parking_lot::Mutex;
use thiserror::Error;
use uuid::Uuid;

const CRC32_BITS: u32 = 32;
const TS_MS_BITS: u32 = 42;
const EPOCH_BITS: u32 = 8;
const PTS_BITS: u32 = 40;
const VERSION_BITS: u32 = 4;
const VARIANT_BITS: u32 = 2;

const TS_MS_MASK: u64 = (1u64 << TS_MS_BITS) - 1;
const PTS_MASK: u64 = (1u64 << PTS_BITS) - 1;

const TS_MS_HI_BITS: u32 = 16;
const TS_MS_MID_BITS: u32 = 12;
const TS_MS_LO_BITS: u32 = 14;
const _: () = assert!(TS_MS_HI_BITS + TS_MS_MID_BITS + TS_MS_LO_BITS == TS_MS_BITS);

const TS_MS_HI_MASK: u64 = (1u64 << TS_MS_HI_BITS) - 1;
const TS_MS_MID_MASK: u64 = (1u64 << TS_MS_MID_BITS) - 1;
const TS_MS_LO_MASK: u64 = (1u64 << TS_MS_LO_BITS) - 1;

const PTS_SHIFT: u32 = 0;
const EPOCH_SHIFT: u32 = PTS_BITS;
const TS_MS_LO_SHIFT: u32 = 48;
const VARIANT_SHIFT: u32 = 62;
const TS_MS_MID_SHIFT: u32 = 64;
const VERSION_SHIFT: u32 = 76;
const TS_MS_HI_SHIFT: u32 = 80;
const CRC32_SHIFT: u32 = 96;

/// UUIDv8 — RFC 9562 §5.8.
pub const UUID_VERSION: u8 = 8;
/// RFC 4122 variant.
pub const UUID_VARIANT: u8 = 0b10;

const VERSION_VAL: u128 = UUID_VERSION as u128;
const VARIANT_VAL: u128 = UUID_VARIANT as u128;

const _: () = assert!(
    CRC32_BITS + TS_MS_BITS + EPOCH_BITS + PTS_BITS + VERSION_BITS + VARIANT_BITS == 128
);

#[inline]
fn pack_ts_ms(ts_ms: u64) -> u128 {
    let hi = ((ts_ms >> (TS_MS_MID_BITS + TS_MS_LO_BITS)) & TS_MS_HI_MASK) as u128;
    let mid = ((ts_ms >> TS_MS_LO_BITS) & TS_MS_MID_MASK) as u128;
    let lo = (ts_ms & TS_MS_LO_MASK) as u128;
    (hi << TS_MS_HI_SHIFT) | (mid << TS_MS_MID_SHIFT) | (lo << TS_MS_LO_SHIFT)
}

#[inline]
fn unpack_ts_ms(v: u128) -> u64 {
    let hi = ((v >> TS_MS_HI_SHIFT) as u64) & TS_MS_HI_MASK;
    let mid = ((v >> TS_MS_MID_SHIFT) as u64) & TS_MS_MID_MASK;
    let lo = ((v >> TS_MS_LO_SHIFT) as u64) & TS_MS_LO_MASK;
    (hi << (TS_MS_MID_BITS + TS_MS_LO_BITS)) | (mid << TS_MS_LO_BITS) | lo
}

/// Default PTS-reset threshold in raw pts units. A keyframe whose
/// PTS regresses by more than this from the previous keyframe is
/// treated as a reset and bumps `epoch`. The default is conservative:
/// 1 s at µs timebase, 1000 s at ms timebase — far past any
/// reordering buffer's reach.
pub const DEFAULT_PTS_RESET_THRESHOLD: i64 = 1_000_000;

/// Default maximum number of distinct sources tracked by a generator
/// before the least-recently-used entry is evicted.
pub const DEFAULT_SOURCE_CAPACITY: usize = 4096;

#[derive(Debug, Error)]
pub enum VideoIdError {
    #[error("ts_ms component overflows 42 bits: {0}")]
    TsMsOverflow(u64),
    #[error("ts_ms component underflows when shifted by {offset_ms} ms")]
    TsMsUnderflow { offset_ms: i64 },
    #[error("ts_ms component overflows when shifted by {offset_ms} ms")]
    TsMsAddOverflow { offset_ms: i64 },
}

/// Compact 128-bit composite frame identifier. See module-level docs
/// for the bit layout and ordering semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VideoId(u128);

impl VideoId {
    pub fn from_parts(
        crc32: u32,
        ts_ms: u64,
        epoch: u8,
        pts: u64,
    ) -> Result<Self, VideoIdError> {
        if ts_ms > TS_MS_MASK {
            return Err(VideoIdError::TsMsOverflow(ts_ms));
        }
        Ok(Self::from_parts_masked(crc32, ts_ms, epoch, pts))
    }

    pub fn from_parts_masked(crc32: u32, ts_ms: u64, epoch: u8, pts: u64) -> Self {
        let v = ((crc32 as u128) << CRC32_SHIFT)
            | pack_ts_ms(ts_ms & TS_MS_MASK)
            | (VERSION_VAL << VERSION_SHIFT)
            | (VARIANT_VAL << VARIANT_SHIFT)
            | ((epoch as u128) << EPOCH_SHIFT)
            | (((pts & PTS_MASK) as u128) << PTS_SHIFT);
        Self(v)
    }

    pub fn as_u128(&self) -> u128 {
        self.0
    }

    pub fn from_u128(v: u128) -> Self {
        Self(v)
    }

    pub fn as_uuid(&self) -> Uuid {
        Uuid::from_u128(self.0)
    }

    pub fn from_uuid(u: Uuid) -> Self {
        Self(u.as_u128())
    }

    pub fn crc32(&self) -> u32 {
        (self.0 >> CRC32_SHIFT) as u32
    }

    pub fn ts_ms(&self) -> u64 {
        unpack_ts_ms(self.0)
    }

    pub fn epoch(&self) -> u8 {
        (self.0 >> EPOCH_SHIFT) as u8
    }

    pub fn pts(&self) -> u64 {
        (self.0 as u64) & PTS_MASK
    }

    /// UUID version nibble — always [`UUID_VERSION`] (= 8) for ids
    /// minted by this module.
    pub fn version(&self) -> u8 {
        ((self.0 >> VERSION_SHIFT) as u8) & 0x0F
    }

    /// UUID variant bits — always [`UUID_VARIANT`] (= `0b10`) for ids
    /// minted by this module (RFC 4122 variant).
    pub fn variant(&self) -> u8 {
        ((self.0 >> VARIANT_SHIFT) as u8) & 0x03
    }

    /// Inclusive lower bound for a wall-clock-time range scan over
    /// frames of `source_id` at `ts_ms`. Pair with [`Self::upper_bound`].
    pub fn lower_bound(source_id: &str, ts_ms: u64) -> Self {
        Self::from_parts_masked(crc32_of(source_id), ts_ms, 0, 0)
    }

    /// Inclusive upper bound for a wall-clock-time range scan over
    /// frames of `source_id` at `ts_ms`.
    pub fn upper_bound(source_id: &str, ts_ms: u64) -> Self {
        Self::from_parts_masked(crc32_of(source_id), ts_ms, u8::MAX, PTS_MASK)
    }

    /// Return a copy of `self` with `ts_ms` shifted by `offset_ms`.
    /// `crc32`, `epoch`, and `pts` are preserved unchanged.
    ///
    /// This is the deterministic migration target for
    /// [`crate::utils::uuid_v7::relative_time_uuid_v7`]. Unlike the
    /// UUIDv7 variant, the result is a pure function of the input —
    /// no fresh entropy is mixed in.
    ///
    /// For constructing range bounds use [`Self::lower_bound`] /
    /// [`Self::upper_bound`] instead; they zero / max-out `epoch`
    /// and `pts` so the returned id sorts at the extreme of the
    /// requested `ts_ms` rather than carrying over the source frame's
    /// pts.
    pub fn shift_time(&self, offset_ms: i64) -> Result<Self, VideoIdError> {
        let ts = self.ts_ms() as i64;
        let new_ts = ts
            .checked_add(offset_ms)
            .ok_or(VideoIdError::TsMsAddOverflow { offset_ms })?;
        if new_ts < 0 {
            return Err(VideoIdError::TsMsUnderflow { offset_ms });
        }
        Self::from_parts(self.crc32(), new_ts as u64, self.epoch(), self.pts())
    }
}

/// Free-function form of [`VideoId::shift_time`], named to mirror
/// [`crate::utils::uuid_v7::relative_time_uuid_v7`] so migrated call
/// sites read the same way.
pub fn relative_time_video_id(id: VideoId, offset_ms: i64) -> Result<VideoId, VideoIdError> {
    id.shift_time(offset_ms)
}

impl From<VideoId> for u128 {
    fn from(id: VideoId) -> u128 {
        id.0
    }
}

impl From<u128> for VideoId {
    fn from(v: u128) -> Self {
        Self(v)
    }
}

impl From<VideoId> for Uuid {
    fn from(id: VideoId) -> Uuid {
        Uuid::from_u128(id.0)
    }
}

impl From<Uuid> for VideoId {
    fn from(u: Uuid) -> Self {
        Self(u.as_u128())
    }
}

pub fn crc32_of(source_id: &str) -> u32 {
    let mut h = Crc32::new();
    h.update(source_id.as_bytes());
    h.finalize()
}

#[derive(Debug, Clone)]
struct SourceState {
    crc32: u32,
    last_kf_ts_ms: u64,
    last_kf_pts: i64,
    epoch: u8,
    current_kf_ts_ms: u64,
    has_keyframe: bool,
}

/// Stateful generator. Holds per-source keyframe-anchor and
/// PTS-reset tracking. Mint one per process at the demuxer/parser
/// stage and route frames through it.
///
/// Per-source state is held in an LRU bounded by
/// [`DEFAULT_SOURCE_CAPACITY`] (configurable). Sources idle for long
/// enough get evicted; a re-emerging source restarts its
/// keyframe-anchor state machine, which is the same behaviour as
/// after an explicit [`Self::forget`].
pub struct VideoIdGenerator {
    sources: LruCache<String, SourceState>,
    pts_reset_threshold: i64,
}

impl Default for VideoIdGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl VideoIdGenerator {
    pub fn new() -> Self {
        Self::with_capacity_and_threshold(DEFAULT_SOURCE_CAPACITY, DEFAULT_PTS_RESET_THRESHOLD)
    }

    pub fn with_threshold(pts_reset_threshold: i64) -> Self {
        Self::with_capacity_and_threshold(DEFAULT_SOURCE_CAPACITY, pts_reset_threshold)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_threshold(capacity, DEFAULT_PTS_RESET_THRESHOLD)
    }

    pub fn with_capacity_and_threshold(capacity: usize, pts_reset_threshold: i64) -> Self {
        let cap = NonZeroUsize::new(capacity).expect("VideoIdGenerator capacity must be > 0");
        Self {
            sources: LruCache::new(cap),
            pts_reset_threshold,
        }
    }

    /// Mint a [`VideoId`] for one frame.
    ///
    /// `wall_clock_ms` is the demuxer's current wall clock in
    /// milliseconds. Callers should derive it from a monotonic-clock
    /// baseline so NTP step-backs cannot stall `ts_ms`.
    ///
    /// `is_keyframe = true` updates the per-source keyframe anchor
    /// and may bump `epoch` on detected PTS reset. Frames received
    /// before the first keyframe of a source anchor on `wall_clock_ms`
    /// so they at least sort by arrival until a keyframe arrives.
    pub fn mint(
        &mut self,
        source_id: &str,
        pts: i64,
        is_keyframe: bool,
        wall_clock_ms: u64,
    ) -> VideoId {
        let state = self
            .sources
            .get_or_insert_mut(source_id.to_string(), || SourceState {
                crc32: crc32_of(source_id),
                last_kf_ts_ms: 0,
                last_kf_pts: 0,
                epoch: 0,
                current_kf_ts_ms: 0,
                has_keyframe: false,
            });

        if is_keyframe {
            let raw = wall_clock_ms & TS_MS_MASK;
            let ts = if state.has_keyframe && raw <= state.last_kf_ts_ms {
                state.last_kf_ts_ms.saturating_add(1) & TS_MS_MASK
            } else {
                raw
            };
            if state.has_keyframe
                && pts < state.last_kf_pts.saturating_sub(self.pts_reset_threshold)
            {
                state.epoch = state.epoch.wrapping_add(1);
            }
            state.last_kf_ts_ms = ts;
            state.last_kf_pts = pts;
            state.current_kf_ts_ms = ts;
            state.has_keyframe = true;
        } else if !state.has_keyframe {
            state.current_kf_ts_ms = wall_clock_ms & TS_MS_MASK;
        }

        VideoId::from_parts_masked(
            state.crc32,
            state.current_kf_ts_ms,
            state.epoch,
            pts as u64,
        )
    }

    /// Drop per-source state. Subsequent frames from the same source
    /// restart the keyframe-anchor state machine.
    pub fn forget(&mut self, source_id: &str) {
        self.sources.pop(source_id);
    }

    pub fn tracked_source_count(&self) -> usize {
        self.sources.len()
    }

    pub fn capacity(&self) -> usize {
        self.sources.cap().get()
    }
}

lazy_static! {
    static ref GLOBAL_GENERATOR: Mutex<VideoIdGenerator> =
        Mutex::new(VideoIdGenerator::new());
}

/// Mint a [`VideoId`] using the process-global generator. Mirror of
/// [`crate::utils::uuid_v7::incremental_uuid_v7`] for migrated
/// call sites that don't carry a generator handle.
pub fn mint(source_id: &str, pts: i64, is_keyframe: bool, wall_clock_ms: u64) -> VideoId {
    GLOBAL_GENERATOR
        .lock()
        .mint(source_id, pts, is_keyframe, wall_clock_ms)
}

pub fn forget(source_id: &str) {
    GLOBAL_GENERATOR.lock().forget(source_id);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parts_round_trip() {
        // ts_ms picked to exercise all three slices (hi/mid/lo) with
        // distinctive non-zero bits.
        let ts_ms = 0x2A_BCDE_F012u64;
        assert!(ts_ms <= TS_MS_MASK);
        let id = VideoId::from_parts(0xDEAD_BEEF, ts_ms, 0xCD, 0xABCDEF1234).unwrap();
        assert_eq!(id.crc32(), 0xDEAD_BEEF);
        assert_eq!(id.ts_ms(), ts_ms);
        assert_eq!(id.epoch(), 0xCD);
        assert_eq!(id.pts(), 0xABCDEF1234);
    }

    #[test]
    fn ts_ms_max_value_round_trips() {
        let id = VideoId::from_parts(0, TS_MS_MASK, 0, 0).unwrap();
        assert_eq!(id.ts_ms(), TS_MS_MASK);
    }

    #[test]
    fn ts_ms_overflow_rejected() {
        let too_big = 1u64 << TS_MS_BITS;
        assert!(matches!(
            VideoId::from_parts(0, too_big, 0, 0),
            Err(VideoIdError::TsMsOverflow(_))
        ));
    }

    #[test]
    fn ids_have_uuidv8_version_and_variant() {
        let mut g = VideoIdGenerator::new();
        let id = g.mint("cam", 42, true, 1_700_000_000_000);
        assert_eq!(id.version(), UUID_VERSION);
        assert_eq!(id.variant(), UUID_VARIANT);
        // Strict-validation cross-check via the uuid crate.
        let u: Uuid = id.into();
        assert_eq!(u.get_version_num() as u8, UUID_VERSION);
        assert!(matches!(u.get_variant(), uuid::Variant::RFC4122));
    }

    #[test]
    fn version_and_variant_constant_across_constructors() {
        let mut g = VideoIdGenerator::new();
        let minted = g.mint("cam", 1, true, 1_700_000_000_000);
        let lo = VideoId::lower_bound("cam", 1_700_000_000_000);
        let hi = VideoId::upper_bound("cam", 1_700_000_000_000);
        let parts = VideoId::from_parts(0, TS_MS_MASK, 0xFF, PTS_MASK).unwrap();
        for id in [minted, lo, hi, parts] {
            assert_eq!(id.version(), UUID_VERSION);
            assert_eq!(id.variant(), UUID_VARIANT);
        }
    }

    #[test]
    fn within_gop_sorts_by_pts() {
        let mut g = VideoIdGenerator::new();
        let i_frame = g.mint("cam", 0, true, 1_000_000);
        let p_frame = g.mint("cam", 100, false, 1_000_001);
        let b_frame = g.mint("cam", 50, false, 1_000_002);
        assert!(i_frame.as_u128() < b_frame.as_u128());
        assert!(b_frame.as_u128() < p_frame.as_u128());
    }

    #[test]
    fn cross_gop_sorts_by_keyframe_arrival() {
        let mut g = VideoIdGenerator::new();
        let g1_i = g.mint("cam", 0, true, 100);
        let g1_p = g.mint("cam", 100, false, 101);
        let g2_i = g.mint("cam", 200, true, 200);
        assert!(g1_i.as_u128() < g1_p.as_u128());
        assert!(g1_p.as_u128() < g2_i.as_u128());
    }

    #[test]
    fn pts_reset_bumps_epoch() {
        let mut g = VideoIdGenerator::with_threshold(1000);
        let pre = g.mint("cam", 100_000, true, 1000);
        let post = g.mint("cam", 0, true, 2000);
        assert_eq!(pre.epoch(), 0);
        assert_eq!(post.epoch(), 1);
        assert!(pre.as_u128() < post.as_u128());
    }

    #[test]
    fn small_backward_pts_does_not_bump_epoch() {
        let mut g = VideoIdGenerator::with_threshold(1000);
        let _kf1 = g.mint("cam", 5000, true, 1000);
        let kf2 = g.mint("cam", 4500, true, 2000);
        assert_eq!(kf2.epoch(), 0);
    }

    #[test]
    fn ts_ms_strict_monotonic_on_tie() {
        let mut g = VideoIdGenerator::new();
        let kf1 = g.mint("cam", 0, true, 100);
        let kf2 = g.mint("cam", 100_000, true, 100);
        assert!(kf2.ts_ms() > kf1.ts_ms());
    }

    #[test]
    fn ts_ms_strict_monotonic_on_backward_clock() {
        let mut g = VideoIdGenerator::new();
        let kf1 = g.mint("cam", 0, true, 1000);
        let kf2 = g.mint("cam", 100_000, true, 500);
        assert!(kf2.ts_ms() > kf1.ts_ms());
    }

    #[test]
    fn wall_clock_bound_orders_correctly() {
        let lo = VideoId::lower_bound("cam", 1000);
        let hi = VideoId::upper_bound("cam", 1000);
        let mut g = VideoIdGenerator::new();
        let id = g.mint("cam", 50, true, 1000);
        assert!(lo.as_u128() <= id.as_u128());
        assert!(id.as_u128() <= hi.as_u128());
    }

    #[test]
    fn shift_time_round_trip() {
        let id = VideoId::lower_bound("cam", 10_000);
        let later = id.shift_time(500).unwrap();
        assert_eq!(later.ts_ms(), 10_500);
        assert_eq!(later.crc32(), id.crc32());
        assert_eq!(later.epoch(), id.epoch());
        assert_eq!(later.pts(), id.pts());
        let earlier = id.shift_time(-1000).unwrap();
        assert_eq!(earlier.ts_ms(), 9_000);
    }

    #[test]
    fn shift_underflow_errors() {
        let id = VideoId::lower_bound("cam", 100);
        assert!(matches!(
            id.shift_time(-1000),
            Err(VideoIdError::TsMsUnderflow { .. })
        ));
    }

    #[test]
    fn shift_preserves_non_time_components() {
        let id = VideoId::from_parts(0xDEAD_BEEF, 10_000, 7, 0xABCDEF1234).unwrap();
        let shifted = id.shift_time(250).unwrap();
        assert_eq!(shifted.crc32(), id.crc32());
        assert_eq!(shifted.epoch(), id.epoch());
        assert_eq!(shifted.pts(), id.pts());
        assert_eq!(shifted.version(), UUID_VERSION);
        assert_eq!(shifted.variant(), UUID_VARIANT);
        assert_eq!(shifted.ts_ms(), 10_250);
    }

    #[test]
    fn relative_time_free_fn_matches_method() {
        let id = VideoId::from_parts(0xDEAD_BEEF, 10_000, 7, 0xABCDEF1234).unwrap();
        let via_method = id.shift_time(250).unwrap();
        let via_free = relative_time_video_id(id, 250).unwrap();
        assert_eq!(via_method, via_free);
    }

    #[test]
    fn relative_time_is_deterministic() {
        let id = VideoId::from_parts(0xDEAD_BEEF, 10_000, 7, 0xABCDEF1234).unwrap();
        let a = relative_time_video_id(id, 1000).unwrap();
        let b = relative_time_video_id(id, 1000).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn forget_resets_source_state() {
        let mut g = VideoIdGenerator::new();
        g.mint("cam", 0, true, 100);
        assert_eq!(g.tracked_source_count(), 1);
        g.forget("cam");
        assert_eq!(g.tracked_source_count(), 0);
    }

    #[test]
    fn lru_evicts_oldest_when_capacity_exceeded() {
        let mut g = VideoIdGenerator::with_capacity_and_threshold(2, DEFAULT_PTS_RESET_THRESHOLD);
        g.mint("cam-a", 0, true, 1000);
        g.mint("cam-b", 0, true, 1001);
        g.mint("cam-c", 0, true, 1002);
        assert_eq!(g.tracked_source_count(), 2);
        assert_eq!(g.capacity(), 2);
    }

    #[test]
    fn default_capacity_is_4096() {
        let g = VideoIdGenerator::new();
        assert_eq!(g.capacity(), 4096);
    }

    #[test]
    fn distinct_sources_have_distinct_crc32_prefixes() {
        let a = crc32_of("cam-a");
        let b = crc32_of("cam-b");
        assert_ne!(a, b);
    }

    #[test]
    fn uuid_round_trip() {
        let id = VideoId::from_parts(0x1122_3344, 0x2A_BCDE_F012, 0x42, 0xABCDEF1234).unwrap();
        let u: Uuid = id.into();
        let back = VideoId::from(u);
        assert_eq!(id, back);
    }

    #[test]
    fn pts_within_40_bits_preserved() {
        let pts = (1u64 << 40) - 1;
        let id = VideoId::from_parts_masked(0, 0, 0, pts);
        assert_eq!(id.pts(), pts);
    }

    #[test]
    fn pts_above_40_bits_masks() {
        let id = VideoId::from_parts_masked(0, 0, 0, (1u64 << 41) | 7);
        assert_eq!(id.pts(), 7);
    }

    #[test]
    fn pre_first_keyframe_anchors_on_arrival() {
        let mut g = VideoIdGenerator::new();
        let p = g.mint("cam", 50, false, 1000);
        let q = g.mint("cam", 60, false, 1100);
        let kf = g.mint("cam", 0, true, 1200);
        assert!(p.as_u128() < q.as_u128());
        assert!(q.as_u128() < kf.as_u128());
    }

    #[test]
    fn distinct_sources_do_not_share_state() {
        let mut g = VideoIdGenerator::new();
        let a = g.mint("cam-a", 0, true, 1000);
        let b = g.mint("cam-b", 0, true, 1000);
        assert_ne!(a.crc32(), b.crc32());
        assert_eq!(a.ts_ms(), b.ts_ms());
        assert_eq!(a.epoch(), 0);
        assert_eq!(b.epoch(), 0);
    }

    #[test]
    fn global_mint_is_consistent_with_generator() {
        forget("global-test-source");
        let g1 = mint("global-test-source", 0, true, 5000);
        let g2 = mint("global-test-source", 100, false, 5001);
        assert!(g1.as_u128() < g2.as_u128());
        assert_eq!(g1.ts_ms(), g2.ts_ms());
        forget("global-test-source");
    }
}
