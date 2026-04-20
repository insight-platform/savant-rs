//! TensorRT engine cache helpers.
//!
//! DeepStream writes a freshly built `.engine` file next to the ONNX model on
//! the first `nvinfer` instantiation.  Callers who point `model-engine-file`
//! at a separate cache directory typically want to **move** that auto-built
//! engine into the cache after the first run so subsequent runs reuse it.
//!
//! [`promote_built_engine`] is the single, idempotent helper for that
//! operation.  It is used by the nvinfer integration tests and by the
//! `savant_samples` crate to keep engine-cache management in one place.

use log::info;
use std::fs;
use std::io;
use std::path::Path;

/// Move a TensorRT engine file from `from` to `to`.
///
/// Behaviour:
///
/// - Returns `Ok(false)` and is a no-op when `from` does not exist or when
///   `to` already exists — making the helper idempotent across repeated
///   runs and safe to call unconditionally after `NvInfer` construction.
/// - Attempts [`fs::rename`] first; if that fails (typical for moves across
///   filesystems) it falls back to [`fs::copy`] followed by a best-effort
///   [`fs::remove_file`] of the source.
/// - On success emits a single `log::info!` line and returns `Ok(true)`.
///
/// # Errors
///
/// Returns [`io::Error`] if both `rename` and `copy` fail.
pub fn promote_built_engine(from: &Path, to: &Path) -> io::Result<bool> {
    if !from.exists() {
        return Ok(false);
    }
    if to.exists() {
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

    #[test]
    fn noop_when_dest_already_present() {
        let dir = tempdir();
        let from = dir.join("built.engine");
        let to = dir.join("cache").join("built.engine");
        fs::create_dir_all(to.parent().unwrap()).unwrap();
        fs::write(&from, b"fresh").unwrap();
        fs::write(&to, b"cached").unwrap();
        let moved = promote_built_engine(&from, &to).unwrap();
        assert!(!moved);
        assert!(from.exists(), "source must be preserved if dest existed");
        assert_eq!(fs::read(&to).unwrap(), b"cached");
        let _ = fs::remove_dir_all(&dir);
    }
}
