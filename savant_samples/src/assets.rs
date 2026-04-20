//! Shared asset-path helpers for the samples crate.
//!
//! All samples locate upstream ONNX / YAML assets (shipped by the
//! `savant_deepstream/*` crates) through [`upstream_asset_path`] and cache
//! TensorRT engines produced on first run under
//! `savant_samples/assets/cache/<model_name>/` through
//! [`engine_cache_dir`].  Keeping these lookups in one place avoids
//! error-prone literal duplication across sample modules and lets us change
//! the cache layout without touching every sample.
//!
//! The actual move-from-auto-path-to-cache step is performed by
//! [`deepstream_nvinfer::engine_cache::promote_built_engine`] — this module
//! only owns the cache-directory layout.

use anyhow::{anyhow, Context, Result};
use std::path::PathBuf;

/// Absolute path to the `savant_samples` crate directory.
pub fn crate_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Absolute path to the workspace root (the parent of
/// [`crate_dir`]).
pub fn workspace_root() -> Result<PathBuf> {
    let dir = crate_dir();
    dir.parent()
        .map(|p| p.to_path_buf())
        .ok_or_else(|| anyhow!("failed to resolve workspace root from {}", dir.display()))
}

/// Canonicalised path to the crate-local `assets/` directory.
pub fn assets_dir() -> PathBuf {
    crate_dir().join("assets")
}

/// Locate an upstream asset shipped by another workspace crate.
///
/// `rel` is a workspace-relative path (e.g.
/// `savant_deepstream/nvinfer/assets/yolo11n.onnx`).  The function looks up
/// the asset relative to the workspace root first, falling back to a
/// sibling-crate path resolved from this crate's manifest directory so the
/// helper works even when the crate is relocated.
///
/// Returns `Err` with the list of attempted locations if the file is absent.
pub fn upstream_asset_path(rel: &str) -> Result<PathBuf> {
    let workspace = workspace_root()?;
    let candidates = [workspace.join(rel), crate_dir().join("..").join(rel)];
    for candidate in &candidates {
        if candidate.is_file() {
            return candidate
                .canonicalize()
                .with_context(|| format!("canonicalize asset path {}", candidate.display()));
        }
    }
    Err(anyhow!(
        "unable to locate upstream asset `{rel}`; tried: {}",
        candidates
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

/// Per-model TensorRT engine cache directory.
///
/// Path: `savant_samples/assets/cache/<model_name>/` — created on first call.
/// Subsequent nvinfer invocations point `model-engine-file` here so the
/// serialized engine produced by the first run is reused; callers combine
/// this directory with [`deepstream_nvinfer::engine_cache::promote_built_engine`]
/// to seed the cache after the first run.
pub fn engine_cache_dir(model_name: &str) -> Result<PathBuf> {
    let dir = assets_dir().join("cache").join(model_name);
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("create engine cache directory {}", dir.display()))?;
    Ok(dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upstream_asset_path_finds_yolo_onnx_and_rejects_missing() {
        let ok = upstream_asset_path("savant_deepstream/nvinfer/assets/yolo11n.onnx")
            .expect("yolo11n.onnx should be shipped by savant_deepstream/nvinfer");
        assert!(ok.is_file(), "{}", ok.display());

        let err = upstream_asset_path("does/not/exist.bin").unwrap_err();
        assert!(err.to_string().contains("unable to locate"));
    }
}
