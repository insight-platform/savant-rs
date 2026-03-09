//! Policy for clearing [`NvDsObjectMeta`] entries around inference.

/// Controls when object metadata is erased from the batch buffer.
///
/// DeepStream secondary mode (`process-mode=2`) reads ROIs from
/// `NvDsObjectMeta` entries attached to the batch meta.  The policy governs
/// when synthetic ROI objects (or stale upstream objects) are removed:
///
/// | Variant | Before submit | After output extracted |
/// |---------|---------------|------------------------|
/// | `None`  | –             | –                      |
/// | `Before`| ✓ (default)   | –                      |
/// | `After` | –             | ✓                      |
/// | `Both`  | ✓             | ✓                      |
///
/// ## Before
/// Clears any existing `NvDsObjectMeta` entries from the **input** buffer
/// before the new ROI objects are attached.  This prevents stale objects
/// from upstream pipeline stages from reaching `nvinfer`.
///
/// ## After
/// When the [`BatchInferenceOutput`](crate::output::BatchInferenceOutput) is
/// **dropped**, all `NvDsObjectMeta` entries are removed from the **output**
/// buffer's frames.  This keeps the buffer clean for any downstream consumers
/// and returns objects to the pool.
///
/// *Safety note*: [`TensorView`](crate::output::TensorView) stores raw
/// pointers into `NvDsInferTensorMeta`.  Those pointers are invalidated when
/// the objects are cleared.  Always consume (or drop) every `TensorView`
/// *before* dropping the owning `BatchInferenceOutput`.
///
/// ## Both
/// Combines `Before` and `After`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MetaClearPolicy {
    /// Do not clear object metas automatically.
    None,
    /// Clear existing objects from the input buffer before attaching ROI
    /// objects. This is the default behaviour.
    #[default]
    Before,
    /// Clear all objects from the output buffer when the
    /// `BatchInferenceOutput` is dropped.
    After,
    /// Clear before submission **and** after the output is dropped.
    Both,
}

impl MetaClearPolicy {
    /// Returns `true` if the policy requires clearing **before** submission.
    #[inline]
    pub fn clear_before(self) -> bool {
        matches!(self, MetaClearPolicy::Before | MetaClearPolicy::Both)
    }

    /// Returns `true` if the policy requires clearing **after** output is
    /// consumed (on `BatchInferenceOutput::drop`).
    #[inline]
    pub fn clear_after(self) -> bool {
        matches!(self, MetaClearPolicy::After | MetaClearPolicy::Both)
    }
}
