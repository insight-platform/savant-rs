//! TensorRT engine cache helpers.
//!
//! DeepStream writes a freshly built `.engine` file next to the ONNX model on
//! the first `nvinfer` instantiation — or whenever it rejected the engine
//! pointed to by `model-engine-file` (e.g. because of a TRT
//! `Platform specific tag mismatch` when the plan was built for a
//! different GPU architecture / TRT version).  Callers who point
//! `model-engine-file` at a separate cache directory typically want to
//! **move** that auto-built engine into the cache after the first run so
//! subsequent runs reuse it.
//!
//! [`promote_built_engine`] is the single helper for that operation.  It
//! is used by the nvinfer integration tests and by the `savant_samples`
//! crate to keep engine-cache management in one place.
//!
//! # Replacement semantics
//!
//! The helper **replaces** the destination when the source exists.  The
//! source (auto-built plan next to the ONNX) is only written by DeepStream
//! when it failed to load whatever was at `model-engine-file` — so an
//! existing destination at that moment is, by construction, stale or
//! incompatible.  Leaving it in place (as the previous idempotent
//! `noop-if-dest-exists` behaviour did) caused every run to rebuild the
//! engine because the stale plan was never overwritten.  The helper is
//! still safe to call unconditionally: when the source is absent (the
//! expected steady state) it is a no-op.

use log::info;
use std::fs;
use std::io;
use std::path::Path;

/// Move a TensorRT engine file from `from` to `to`, replacing the
/// destination if it already exists.
///
/// Behaviour:
///
/// - Returns `Ok(false)` and is a no-op when `from` does not exist — the
///   expected steady state once the cache has been seeded and a later
///   DeepStream run successfully loaded the cached plan.
/// - When `from` exists, attempts [`fs::rename`] first — which atomically
///   replaces `to` on POSIX; if that fails (typical for moves across
///   filesystems) it falls back to [`fs::copy`] followed by a
///   best-effort [`fs::remove_file`] of the source.  Either way any
///   pre-existing `to` is overwritten.
/// - On success emits a single `log::info!` line and returns `Ok(true)`.
///
/// # Errors
///
/// Returns [`io::Error`] if both `rename` and `copy` fail.
pub fn promote_built_engine(from: &Path, to: &Path) -> io::Result<bool> {
    if !from.exists() {
        return Ok(false);
    }
    if let Some(parent) = to.parent() {
        fs::create_dir_all(parent)?;
    }
    match fs::rename(from, to) {
        Ok(()) => {
            info!(
                "promoted TensorRT engine to cache: {} -> {}",
                from.display(),
                to.display()
            );
            Ok(true)
        }
        Err(_) => {
            fs::copy(from, to)?;
            let _ = fs::remove_file(from);
            info!(
                "promoted TensorRT engine to cache (via copy): {} -> {}",
                from.display(),
                to.display()
            );
            Ok(true)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tempdir() -> std::path::PathBuf {
        let base =
            std::env::temp_dir().join(format!("nvinfer-engine-cache-ut-{}", std::process::id()));
        fs::create_dir_all(&base).expect("create test tempdir");
        base
    }

    #[test]
    fn promotes_when_source_present_and_dest_absent() {
        let dir = tempdir();
        let from = dir.join("built.engine");
        let to = dir.join("cache").join("built.engine");
        {
            let mut f = fs::File::create(&from).unwrap();
            f.write_all(b"hello").unwrap();
        }
        let moved = promote_built_engine(&from, &to).unwrap();
        assert!(moved);
        assert!(!from.exists());
        assert!(to.exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn noop_when_source_missing() {
        let dir = tempdir();
        let from = dir.join("missing.engine");
        let to = dir.join("cache/missing.engine");
        let moved = promote_built_engine(&from, &to).unwrap();
        assert!(!moved);
        assert!(!to.exists());
        let _ = fs::remove_dir_all(&dir);
    }

    /// When DeepStream rebuilds an engine it writes the fresh plan next to
    /// the ONNX — the rebuild itself is the signal that whatever lived at
    /// `to` was rejected (TRT `Platform specific tag mismatch` or a
    /// similar incompatibility).  The helper must therefore overwrite the
    /// stale plan so the next run reuses the freshly-built one; otherwise
    /// the cache keeps serving the same broken plan and every run
    /// re-triggers the multi-minute TRT build.
    #[test]
    fn replaces_dest_when_source_present() {
        let dir = tempdir();
        let from = dir.join("built.engine");
        let to = dir.join("cache").join("built.engine");
        fs::create_dir_all(to.parent().unwrap()).unwrap();
        fs::write(&from, b"fresh").unwrap();
        fs::write(&to, b"cached").unwrap();
        let moved = promote_built_engine(&from, &to).unwrap();
        assert!(moved);
        assert!(
            !from.exists(),
            "source must be moved into cache, not duplicated"
        );
        assert_eq!(
            fs::read(&to).unwrap(),
            b"fresh",
            "stale cached plan must be replaced by the freshly-built one"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// Steady state after the cache has been seeded: DeepStream loads the
    /// cached plan without writing a new one, so the helper must be a
    /// no-op (source absent).
    #[test]
    fn noop_when_source_absent_and_dest_present() {
        let dir = tempdir();
        let from = dir.join("built.engine");
        let to = dir.join("cache").join("built.engine");
        fs::create_dir_all(to.parent().unwrap()).unwrap();
        fs::write(&to, b"cached").unwrap();
        let moved = promote_built_engine(&from, &to).unwrap();
        assert!(!moved);
        assert!(to.exists());
        assert_eq!(fs::read(&to).unwrap(), b"cached");
        let _ = fs::remove_dir_all(&dir);
    }
}
