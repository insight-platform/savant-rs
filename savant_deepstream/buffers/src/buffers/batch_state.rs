use std::time::Instant;

/// Generic in-memory batch accumulator state with an optional submit deadline.
///
/// This is shared by higher-level operators that implement
/// size/time-driven batching policies.
pub struct BatchState<T> {
    /// Accumulated items for the next submission.
    pub frames: Vec<T>,
    /// Optional deadline for timer-driven submission.
    pub deadline: Option<Instant>,
}

impl<T> BatchState<T> {
    /// Create an empty batch state.
    pub fn new() -> Self {
        Self {
            frames: Vec::new(),
            deadline: None,
        }
    }

    /// Drain all queued items and clear the deadline.
    pub fn take(&mut self) -> Vec<T> {
        self.deadline = None;
        std::mem::take(&mut self.frames)
    }

    /// Whether no items are currently queued.
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }
}

impl<T> Default for BatchState<T> {
    fn default() -> Self {
        Self::new()
    }
}
