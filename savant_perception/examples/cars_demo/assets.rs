//! Shared asset-path helpers for the samples crate.
//!
//! All samples locate upstream ONNX / YAML assets (shipped by the
//! `savant_deepstream/*` crates) through [`upstream_asset_path`] and cache
//! TensorRT engines produced on first run under
//! `savant_perception/assets/cache/<model_name>/<platform_tag>/` through
//! [`engine_cache_dir`].  Keeping these lookups in one place avoids
//! error-prone literal duplication across sample modules and lets us change
//! the cache layout without touching every sample.
//!
//! TensorRT plan files are **not portable** across GPU architectures or TRT
//! runtime versions (`IRuntime::deserializeCudaEngine` fails with
//! `Platform specific tag mismatch`).  The cache layout therefore
//! namespaces engines under a platform tag derived from
//! [`nvidia_gpu_utils::gpu_platform_tag`] — e.g. `ada/`, `hopper/`, or
//! `orin_nano_8gb/` — so a cache checked into the repo or shared between
//! machines never serves a plan file built for another platform.  The
//! scheme matches what the `nvinfer` integration tests use
//! (`assets/engines/<platform_tag>/`).
//!
//! The actual move-from-auto-path-to-cache step is performed by
//! [`deepstream_nvinfer::engine_cache::promote_built_engine`] — this module
//! only owns the cache-directory layout.

use anyhow::{anyhow, Context, Result};
use std::path::PathBuf;

/// Absolute path to the `savant_perception` crate directory.
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

/// Platform tag used to namespace the engine cache for a given GPU.
///
/// Delegates to [`nvidia_gpu_utils::gpu_platform_tag`], falling back to
/// `"unknown"` when the tag cannot be determined.  Using this keeps the
/// cache layout aligned with the `nvinfer` integration tests and prevents
/// TRT `Platform specific tag mismatch` deserialization failures when the
/// repository (or a copied workspace) is shared across machines with
/// different GPU architectures or TRT versions.
pub fn engine_cache_platform_tag(gpu_id: u32) -> String {
    nvidia_gpu_utils::gpu_platform_tag(gpu_id).unwrap_or_else(|_| "unknown".to_string())
}

/// Per-model, per-platform TensorRT engine cache directory.
///
/// Path: `savant_perception/assets/cache/<model_name>/<platform_tag>/` —
/// created on first call.  Subsequent nvinfer invocations point
/// `model-engine-file` here so the serialized engine produced by the first
/// run is reused; callers combine this directory with
/// [`deepstream_nvinfer::engine_cache::promote_built_engine`] to seed the
/// cache after the first run.
///
/// The `<platform_tag>` segment is required — see
/// [`engine_cache_platform_tag`].  Without it a plan file built on (say) a
/// laptop RTX 4060 (`ada`) would be rejected on a Jetson Orin
/// (`orin_nx_*`) or on a host with a different TRT runtime version with
/// `Platform specific tag mismatch detected`, forcing DeepStream to
/// rebuild the engine from ONNX on every run — the exact symptom we want
/// the cache to eliminate.  `promote_built_engine` now replaces an
/// incompatible destination on rebuild, but segmenting by platform first
/// means it rarely has to: each machine writes into its own bucket.
pub fn engine_cache_dir(model_name: &str, gpu_id: u32) -> Result<PathBuf> {
    let tag = engine_cache_platform_tag(gpu_id);
    let dir = assets_dir().join("cache").join(model_name).join(&tag);
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

    /// The engine cache path must embed the platform tag so that a plan
    /// built for one GPU / TRT runtime never gets served to another —
    /// otherwise `IRuntime::deserializeCudaEngine` fails with
    /// `Platform specific tag mismatch` and DeepStream rebuilds the
    /// engine every run.
    #[test]
    fn engine_cache_dir_embeds_model_and_platform_segments() {
        let dir = engine_cache_dir("my-model", 0).expect("create engine cache dir");
        let components: Vec<_> = dir
            .components()
            .rev()
            .take(2)
            .map(|c| c.as_os_str().to_string_lossy().into_owned())
            .collect();
        let platform = &components[0];
        let model = &components[1];
        assert_eq!(model, "my-model", "last but one segment must be the model");
        assert!(
            !platform.is_empty(),
            "platform tag segment must not be empty (got {:?})",
            dir
        );
        assert!(
            dir.is_dir(),
            "cache dir must exist on disk: {}",
            dir.display()
        );
    }
}
