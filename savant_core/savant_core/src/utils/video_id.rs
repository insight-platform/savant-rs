//! Composite 128-bit video frame identifier intended to replace
//! UUIDv7-based frame ids during the migration tracked alongside this
//! module.
//!
//! Encoded as a fully RFC 9562 §5.8 (UUIDv8) compliant UUID — the
//! version nibble at bits `[79:76]` is fixed at `0b1000` (= 8) and
//! the variant bits at `[63:62]` are fixed at `0b10`. The remaining
//! 122 application-defined bits hold three fields:
//!
//! ```text
//!  127        80 79  76 75       64 63   62 61   58 57    50 49     0
//! +-------------+------+-----------+------+-------+--------+---------+
//! |   ts_ns hi  | ver  |  ts_ns    | var  | ts_ns | epoch  |   pts   |
//! |   48 bits   | =8   |  mid: 12  | =10  | lo: 4 |  8 b   |  50 b   |
//! +-------------+------+-----------+------+-------+--------+---------+
//! ```
//!
//! * `ts_ns` — wall-clock **nanoseconds** of the GOP's keyframe at
//!   the demuxer. Full 64 bits (≈ 584 years from 1970), stored across
//!   three slices that bracket the fixed version/variant nibbles.
//!   Constant for every frame in the same GOP. Strict-monotonic per
//!   source via the disorder-avoidance pattern used by
//!   `incremental_uuid_v7`.
//! * `epoch` — bumps when the demuxer detects a PTS reset for the
//!   source.
//! * `pts` — frame PTS, low-order tail. 50 bits is enough for any
//!   single GOP at any common timebase.
//!
//! Source identity is **not** encoded in the id. The `source_id` is
//! used purely as the LRU-keyed accounting handle inside the
//! [`VideoIdGenerator`] — different sources with overlapping
//! `(ts_ns, epoch, pts)` will produce identical ids. Every existing
//! consumer in this codebase (meta_merge merge_queue, replay
//! RocksDB keyframe index, pipeline keyframe_tracking) already scopes
//! lookups by `source_id` separately, so this is safe.
//!
//! Sort order on the u128 is `(ts_ns, epoch, pts)` — version and
//! variant are constant and don't perturb comparisons. That preserves
//! every property the design was built around: within-GOP PTS-sort,
//! cross-GOP keyframe-arrival sort, reset tolerance via wall-clock
//! dominance, wall-clock-time keyframe range scans.

use std::num::NonZeroUsize;

use lazy_static::lazy_static;
use lru::LruCache;
use parking_lot::Mutex;
use thiserror::Error;
use uuid::Uuid;

const TS_NS_BITS: u32 = 64;
const EPOCH_BITS: u32 = 8;
const PTS_BITS: u32 = 50;
const VERSION_BITS: u32 = 4;
const VARIANT_BITS: u32 = 2;

const TS_NS_HI_BITS: u32 = 48;
const TS_NS_MID_BITS: u32 = 12;
const TS_NS_LO_BITS: u32 = 4;
const _: () = assert!(TS_NS_HI_BITS + TS_NS_MID_BITS + TS_NS_LO_BITS == TS_NS_BITS);

const PTS_MASK: u64 = (1u64 << PTS_BITS) - 1;
const TS_NS_HI_MASK: u64 = (1u64 << TS_NS_HI_BITS) - 1;
const TS_NS_MID_MASK: u64 = (1u64 << TS_NS_MID_BITS) - 1;
const TS_NS_LO_MASK: u64 = (1u64 << TS_NS_LO_BITS) - 1;

const PTS_SHIFT: u32 = 0;
const EPOCH_SHIFT: u32 = PTS_BITS;
const TS_NS_LO_SHIFT: u32 = 58;
const VARIANT_SHIFT: u32 = 62;
const TS_NS_MID_SHIFT: u32 = 64;
const VERSION_SHIFT: u32 = 76;
const TS_NS_HI_SHIFT: u32 = 80;

/// UUIDv8 — RFC 9562 §5.8.
pub const UUID_VERSION: u8 = 8;
/// RFC 4122 variant.
pub const UUID_VARIANT: u8 = 0b10;

const VERSION_VAL: u128 = UUID_VERSION as u128;
const VARIANT_VAL: u128 = UUID_VARIANT as u128;

const _: () =
    assert!(TS_NS_BITS + EPOCH_BITS + PTS_BITS + VERSION_BITS + VARIANT_BITS == 128);

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
    #[error("pts component overflows {0} bits: {1}")]
    PtsOverflow(u32, u64),
    #[error("ts_ns underflows when shifted by {offset_ms} ms")]
    TsNsUnderflow { offset_ms: i64 },
    #[error("ts_ns overflows when shifted by {offset_ms} ms")]
    TsNsAddOverflow { offset_ms: i64 },
}

#[inline]
fn pack_ts_ns(ts_ns: u64) -> u128 {
    let hi = ((ts_ns >> (TS_NS_MID_BITS + TS_NS_LO_BITS)) & TS_NS_HI_MASK) as u128;
    let mid = ((ts_ns >> TS_NS_LO_BITS) & TS_NS_MID_MASK) as u128;
    let lo = (ts_ns & TS_NS_LO_MASK) as u128;
    (hi << TS_NS_HI_SHIFT) | (mid << TS_NS_MID_SHIFT) | (lo << TS_NS_LO_SHIFT)
}

#[inline]
fn unpack_ts_ns(v: u128) -> u64 {
    let hi = ((v >> TS_NS_HI_SHIFT) as u64) & TS_NS_HI_MASK;
    let mid = ((v >> TS_NS_MID_SHIFT) as u64) & TS_NS_MID_MASK;
    let lo = ((v >> TS_NS_LO_SHIFT) as u64) & TS_NS_LO_MASK;
    (hi << (TS_NS_MID_BITS + TS_NS_LO_BITS)) | (mid << TS_NS_LO_BITS) | lo
}

/// Compact 128-bit composite frame identifier. See module-level docs
/// for the bit layout and ordering semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VideoId(u128);

impl VideoId {
    /// Construct from explicit components. Errors on `pts` overflow;
    /// `ts_ns` and `epoch` always fit in their native widths so they
    /// can't overflow.
    pub fn from_parts(ts_ns: u64, epoch: u8, pts: u64) -> Result<Self, VideoIdError> {
        if pts > PTS_MASK {
            return Err(VideoIdError::PtsOverflow(PTS_BITS, pts));
        }
        Ok(Self::from_parts_masked(ts_ns, epoch, pts))
    }

    /// Construct from explicit components, masking `pts` silently.
    pub fn from_parts_masked(ts_ns: u64, epoch: u8, pts: u64) -> Self {
        let v = pack_ts_ns(ts_ns)
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

    pub fn ts_ns(&self) -> u64 {
        unpack_ts_ns(self.0)
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

    /// Inclusive lower bound of the keyframe range scan at `ts_ns`.
    /// `epoch` and `pts` are zeroed so the result sorts at the bottom
    /// of the requested ts.
    pub fn lower_bound(ts_ns: u64) -> Self {
        Self::from_parts_masked(ts_ns, 0, 0)
    }

    /// Inclusive upper bound of the keyframe range scan at `ts_ns`.
    /// `epoch` and `pts` are maxed so the result sorts at the top of
    /// the requested ts.
    pub fn upper_bound(ts_ns: u64) -> Self {
        Self::from_parts_masked(ts_ns, u8::MAX, PTS_MASK)
    }

    /// Return a copy of `self` with `ts_ns` shifted by `offset_ms`
    /// milliseconds. `epoch` and `pts` are preserved unchanged.
    ///
    /// Migration target for
    /// [`crate::utils::uuid_v7::relative_time_uuid_v7`]. Deterministic
    /// — no fresh entropy is mixed in.
    ///
    /// For range bounds use [`Self::lower_bound`] /
    /// [`Self::upper_bound`] instead — they zero / max-out `epoch`
    /// and `pts` so the returned id sorts at the extreme of the
    /// requested time rather than carrying over the source frame's
    /// pts.
    pub fn shift_time(&self, offset_ms: i64) -> Result<Self, VideoIdError> {
        let offset_ns = (offset_ms as i128).saturating_mul(1_000_000);
        let new_ts_signed = (self.ts_ns() as i128)
            .checked_add(offset_ns)
            .ok_or(VideoIdError::TsNsAddOverflow { offset_ms })?;
        if new_ts_signed < 0 {
            return Err(VideoIdError::TsNsUnderflow { offset_ms });
        }
        if new_ts_signed > u64::MAX as i128 {
            return Err(VideoIdError::TsNsAddOverflow { offset_ms });
        }
        Self::from_parts(new_ts_signed as u64, self.epoch(), self.pts())
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

#[derive(Debug, Clone)]
struct SourceState {
    last_kf_ts_ns: u64,
    last_kf_pts: i64,
    epoch: u8,
    current_kf_ts_ns: u64,
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
    /// `wall_clock_ns` is the demuxer's current wall clock in
    /// nanoseconds. Callers should derive it from a monotonic-clock
    /// baseline so NTP step-backs cannot stall `ts_ns`.
    ///
    /// `is_keyframe = true` updates the per-source keyframe anchor
    /// and may bump `epoch` on detected PTS reset. Frames received
    /// before the first keyframe of a source anchor on `wall_clock_ns`
    /// so they at least sort by arrival until a keyframe arrives.
    pub fn mint(
        &mut self,
        source_id: &str,
        pts: i64,
        is_keyframe: bool,
        wall_clock_ns: u64,
    ) -> VideoId {
        let state = self
            .sources
            .get_or_insert_mut(source_id.to_string(), || SourceState {
                last_kf_ts_ns: 0,
                last_kf_pts: 0,
                epoch: 0,
                current_kf_ts_ns: 0,
                has_keyframe: false,
            });

        if is_keyframe {
            let ts = if state.has_keyframe && wall_clock_ns <= state.last_kf_ts_ns {
                state.last_kf_ts_ns.saturating_add(1)
            } else {
                wall_clock_ns
            };
            if state.has_keyframe
                && pts < state.last_kf_pts.saturating_sub(self.pts_reset_threshold)
            {
                state.epoch = state.epoch.wrapping_add(1);
            }
            state.last_kf_ts_ns = ts;
            state.last_kf_pts = pts;
            state.current_kf_ts_ns = ts;
            state.has_keyframe = true;
        } else if !state.has_keyframe {
            state.current_kf_ts_ns = wall_clock_ns;
        }

        VideoId::from_parts_masked(state.current_kf_ts_ns, state.epoch, pts as u64)
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
pub fn mint(source_id: &str, pts: i64, is_keyframe: bool, wall_clock_ns: u64) -> VideoId {
    GLOBAL_GENERATOR
        .lock()
        .mint(source_id, pts, is_keyframe, wall_clock_ns)
}

pub fn forget(source_id: &str) {
    GLOBAL_GENERATOR.lock().forget(source_id);
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_TS_NS: u64 = 0xABCD_EF01_2345_6789;
    const SAMPLE_PTS: u64 = 0x3_FFFF_FFFF_FFFF; // 50 bits, max-1

    #[test]
    fn parts_round_trip() {
        // ts_ns picked to exercise all three slices (hi/mid/lo) with
        // distinctive non-zero bits in each.
        let id = VideoId::from_parts(SAMPLE_TS_NS, 0xCD, SAMPLE_PTS).unwrap();
        assert_eq!(id.ts_ns(), SAMPLE_TS_NS);
        assert_eq!(id.epoch(), 0xCD);
        assert_eq!(id.pts(), SAMPLE_PTS);
    }

    #[test]
    fn ts_ns_max_value_round_trips() {
        let id = VideoId::from_parts(u64::MAX, 0, 0).unwrap();
        assert_eq!(id.ts_ns(), u64::MAX);
    }

    #[test]
    fn pts_overflow_rejected() {
        let too_big = 1u64 << PTS_BITS;
        assert!(matches!(
            VideoId::from_parts(0, 0, too_big),
            Err(VideoIdError::PtsOverflow(..))
        ));
    }

    #[test]
    fn ids_have_uuidv8_version_and_variant() {
        let mut g = VideoIdGenerator::new();
        let id = g.mint("cam", 42, true, 1_700_000_000_000_000_000);
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
        let minted = g.mint("cam", 1, true, 1_700_000_000_000_000_000);
        let lo = VideoId::lower_bound(1_700_000_000_000_000_000);
        let hi = VideoId::upper_bound(1_700_000_000_000_000_000);
        let parts = VideoId::from_parts(u64::MAX, 0xFF, PTS_MASK).unwrap();
        for id in [minted, lo, hi, parts] {
            assert_eq!(id.version(), UUID_VERSION);
            assert_eq!(id.variant(), UUID_VARIANT);
        }
    }

    #[test]
    fn within_gop_sorts_by_pts() {
        let mut g = VideoIdGenerator::new();
        let i_frame = g.mint("cam", 0, true, 1_000_000_000_000);
        let p_frame = g.mint("cam", 100, false, 1_000_000_000_001);
        let b_frame = g.mint("cam", 50, false, 1_000_000_000_002);
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
    fn ts_ns_strict_monotonic_on_tie() {
        let mut g = VideoIdGenerator::new();
        let kf1 = g.mint("cam", 0, true, 100);
        let kf2 = g.mint("cam", 100_000, true, 100);
        assert!(kf2.ts_ns() > kf1.ts_ns());
    }

    #[test]
    fn ts_ns_strict_monotonic_on_backward_clock() {
        let mut g = VideoIdGenerator::new();
        let kf1 = g.mint("cam", 0, true, 1000);
        let kf2 = g.mint("cam", 100_000, true, 500);
        assert!(kf2.ts_ns() > kf1.ts_ns());
    }

    #[test]
    fn wall_clock_bound_orders_correctly() {
        let lo = VideoId::lower_bound(1_000_000_000);
        let hi = VideoId::upper_bound(1_000_000_000);
        let mut g = VideoIdGenerator::new();
        let id = g.mint("cam", 50, true, 1_000_000_000);
        assert!(lo.as_u128() <= id.as_u128());
        assert!(id.as_u128() <= hi.as_u128());
    }

    #[test]
    fn shift_time_round_trip() {
        let id = VideoId::lower_bound(10_000_000_000);
        let later = id.shift_time(500).unwrap();
        // +500 ms == +500_000_000 ns
        assert_eq!(later.ts_ns(), 10_500_000_000);
        assert_eq!(later.epoch(), id.epoch());
        assert_eq!(later.pts(), id.pts());
        let earlier = id.shift_time(-1000).unwrap();
        assert_eq!(earlier.ts_ns(), 9_000_000_000);
    }

    #[test]
    fn shift_underflow_errors() {
        let id = VideoId::lower_bound(100_000);
        assert!(matches!(
            id.shift_time(-10_000),
            Err(VideoIdError::TsNsUnderflow { .. })
        ));
    }

    #[test]
    fn shift_preserves_non_time_components() {
        let id = VideoId::from_parts(10_000_000_000, 7, 0xABCDEF1234).unwrap();
        let shifted = id.shift_time(250).unwrap();
        assert_eq!(shifted.epoch(), id.epoch());
        assert_eq!(shifted.pts(), id.pts());
        assert_eq!(shifted.version(), UUID_VERSION);
        assert_eq!(shifted.variant(), UUID_VARIANT);
        assert_eq!(shifted.ts_ns(), 10_250_000_000);
    }

    #[test]
    fn relative_time_free_fn_matches_method() {
        let id = VideoId::from_parts(10_000_000_000, 7, 0xABCDEF1234).unwrap();
        let via_method = id.shift_time(250).unwrap();
        let via_free = relative_time_video_id(id, 250).unwrap();
        assert_eq!(via_method, via_free);
    }

    #[test]
    fn relative_time_is_deterministic() {
        let id = VideoId::from_parts(10_000_000_000, 7, 0xABCDEF1234).unwrap();
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
    fn uuid_round_trip() {
        let id = VideoId::from_parts(SAMPLE_TS_NS, 0x42, 0xABCDEF1234).unwrap();
        let u: Uuid = id.into();
        let back = VideoId::from(u);
        assert_eq!(id, back);
    }

    #[test]
    fn pts_at_50_bit_max_round_trips() {
        let id = VideoId::from_parts_masked(0, 0, PTS_MASK);
        assert_eq!(id.pts(), PTS_MASK);
    }

    #[test]
    fn pts_above_50_bits_masks() {
        let id = VideoId::from_parts_masked(0, 0, (1u64 << 51) | 7);
        assert_eq!(id.pts(), 7);
    }

    #[test]
    fn pre_first_keyframe_anchors_on_arrival() {
        let mut g = VideoIdGenerator::new();
        let p = g.mint("cam", 50, false, 1_000_000_000);
        let q = g.mint("cam", 60, false, 1_100_000_000);
        let kf = g.mint("cam", 0, true, 1_200_000_000);
        assert!(p.as_u128() < q.as_u128());
        assert!(q.as_u128() < kf.as_u128());
    }

    #[test]
    fn distinct_sources_with_same_ts_pts_collide() {
        // Source identity is no longer encoded in the id. With identical
        // (ts_ns, epoch, pts), two different sources produce identical
        // ids — by design. Per-source state still keeps each generator
        // accounting separate.
        let mut g = VideoIdGenerator::new();
        let a = g.mint("cam-a", 42, true, 1_000_000_000);
        let b = g.mint("cam-b", 42, true, 1_000_000_000);
        assert_eq!(a, b);
    }

    #[test]
    fn global_mint_is_consistent_with_generator() {
        forget("global-test-source");
        let g1 = mint("global-test-source", 0, true, 5_000_000_000);
        let g2 = mint("global-test-source", 100, false, 5_000_000_001);
        assert!(g1.as_u128() < g2.as_u128());
        assert_eq!(g1.ts_ns(), g2.ts_ns());
        forget("global-test-source");
    }
}
