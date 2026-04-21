//! YOLO-11n model configuration for the `cars_tracking` sample.
//!
//! The sample is frame-by-frame: nvinfer is configured with
//! `max_batch_size = 1` and `max_batch_wait = 0` (see
//! [`build_nvinfer_config`]) because input comes from a file and we want
//! detections available immediately for every decoded frame.
//!
//! To avoid the ~20-second TensorRT build every run, the serialized engine
//! produced on first invocation is cached under
//! `savant_samples/assets/cache/yolo11n/<platform_tag>/` and reused
//! thereafter.  The `<platform_tag>` segment (e.g. `ada`,
//! `orin_nano_8gb`) is required because TensorRT plan files are only
//! valid on the platform they were built on — loading a plan built on a
//! different architecture or TRT version fails with
//! `Platform specific tag mismatch detected`, which forces DeepStream to
//! rebuild the engine from ONNX on every run.  See
//! [`crate::assets::engine_cache_dir`] for the layout rationale.
//!
//! The `model-engine-file` property is set so nvinfer picks up the cached
//! engine when it exists; [`promote_yolo11n_engine`] moves the freshly-
//! built engine into the cache after the first `NvInferBatchingOperator::new`
//! call (replacing any stale plan that was found to be incompatible).

use anyhow::Result;
use deepstream_buffers::VideoFormat;
use deepstream_nvinfer::{
    ModelColorFormat, ModelInputScaling, NvInferBatchingOperatorConfig, NvInferConfig,
};
// `HashMap` here must be `std::collections::HashMap` because it is
// handed to `NvInferConfig::new`, which is a boundary API that takes
// the std type.  The rest of `cars_tracking` uses `hashbrown`.
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::assets;

/// Fixed YOLO input width in pixels.
pub const MODEL_WIDTH: u32 = 640;
/// Fixed YOLO input height in pixels.
pub const MODEL_HEIGHT: u32 = 640;
/// Number of COCO classes expected by YOLO-11n.
pub const YOLO_NUM_CLASSES: usize = 80;
/// Namespace used for attached detection objects.
pub const DETECTION_NAMESPACE: &str = "det";
/// Workspace-relative path to the YOLO11n ONNX asset.
const YOLO_ONNX_REL: &str = "savant_deepstream/nvinfer/assets/yolo11n.onnx";
/// Short model identifier used for the engine-cache subdirectory.
pub const MODEL_NAME: &str = "yolo11n";

/// COCO class IDs we keep post-YOLO (vehicles).
///
/// Mapping:
/// - `2` car
/// - `3` motorbike
/// - `5` bus
/// - `7` truck
pub const VEHICLE_CLASS_IDS: &[usize] = &[2, 3, 5, 7];

/// Look up the human-readable label for a vehicle COCO class id.
///
/// Returns `None` for non-vehicle IDs.
pub fn vehicle_label(class_id: usize) -> Option<&'static str> {
    match class_id {
        2 => Some("car"),
        3 => Some("motorbike"),
        5 => Some("bus"),
        7 => Some("truck"),
        _ => None,
    }
}

/// Locate the `yolo11n.onnx` asset shipped by `savant_deepstream/nvinfer`.
pub fn yolo11n_onnx_path() -> Result<PathBuf> {
    assets::upstream_asset_path(YOLO_ONNX_REL)
}

/// Filename of the TensorRT engine produced by DeepStream for YOLO11n at
/// batch 1, fp16, on the given GPU.
fn engine_filename(gpu_id: u32) -> String {
    format!("yolo11n.onnx_b1_gpu{gpu_id}_fp16.engine")
}

/// Absolute path of the cached TensorRT engine (may not yet exist).
///
/// The returned path lives under the platform-specific cache directory
/// (`savant_samples/assets/cache/yolo11n/<platform_tag>/`) — see
/// [`assets::engine_cache_dir`].
pub fn yolo11n_engine_cache_path(gpu_id: u32) -> Result<PathBuf> {
    Ok(assets::engine_cache_dir(MODEL_NAME, gpu_id)?.join(engine_filename(gpu_id)))
}

/// Build the nvinfer property map for YOLO-11n.
///
/// Jetson (`aarch64`) additionally enables GPU scaling with
/// `scaling-compute-hw=1`.  `model-engine-file` is always set to the cached
/// path so a previously built engine is reused across runs; on first run
/// DeepStream writes the engine next to the ONNX file — [`promote_yolo11n_engine`]
/// relocates it into the cache after the first operator creation.
pub fn yolo11n_properties(
    onnx_path: &Path,
    gpu_id: u32,
    engine_cache_path: &Path,
) -> HashMap<String, String> {
    let props = vec![
        ("onnx-file", onnx_path.to_string_lossy().into_owned()),
        (
            "model-engine-file",
            engine_cache_path.to_string_lossy().into_owned(),
        ),
        ("gpu-id", gpu_id.to_string()),
        ("network-mode", "2".to_string()),
        ("workspace-size", "2048".to_string()),
        ("batch-size", "1".to_string()),
        ("net-scale-factor", "0.003921569790691137".to_string()),
        ("offsets", "0.0;0.0;0.0".to_string()),
        ("output-blob-names", "output0".to_string()),
        ("scaling-compute-hw", "1".to_string()),
    ];
    props.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

/// Build the full `NvInferBatchingOperatorConfig` for the cars-demo sample.
///
/// The operator runs one frame at a time: `max_batch_size = 1` and
/// `max_batch_wait = 0`.
pub fn build_nvinfer_config(gpu_id: u32) -> Result<NvInferBatchingOperatorConfig> {
    let onnx_path = yolo11n_onnx_path()?;
    let engine_cache = yolo11n_engine_cache_path(gpu_id)?;
    let nvinfer = NvInferConfig::new(
        yolo11n_properties(&onnx_path, gpu_id, &engine_cache),
        VideoFormat::RGBA,
        MODEL_WIDTH,
        MODEL_HEIGHT,
        ModelColorFormat::RGB,
    )
    .gpu_id(gpu_id)
    .scaling(ModelInputScaling::KeepAspectRatioSymmetric)
    .name("cars-demo/yolo11n");

    Ok(NvInferBatchingOperatorConfig::builder(nvinfer)
        .max_batch_size(1)
        .max_batch_wait(Duration::from_millis(0))
        .build())
}

/// Move the freshly-built TensorRT engine next to the ONNX file into our
/// per-model cache so subsequent runs reuse it via `model-engine-file`.
///
/// Invoke once after `NvInferBatchingOperator::new` succeeds.  No-op if the
/// engine is already cached (second and later runs) or if DeepStream reused
/// a cached engine without writing a new one.
pub fn promote_yolo11n_engine(gpu_id: u32) -> Result<()> {
    let onnx_path = yolo11n_onnx_path()?;
    let filename = engine_filename(gpu_id);
    let auto_path = onnx_path
        .parent()
        .map(|p| p.join(&filename))
        .unwrap_or_else(|| PathBuf::from(&filename));
    let cache_path = yolo11n_engine_cache_path(gpu_id)?;
    deepstream_nvinfer::engine_cache::promote_built_engine(&auto_path, &cache_path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The COCO class → vehicle-label table is consumed by the draw spec and
    /// the detection attach logic; a silent change here drops detections.
    #[test]
    fn vehicle_label_maps_vehicle_classes_only() {
        assert_eq!(vehicle_label(2), Some("car"));
        assert_eq!(vehicle_label(3), Some("motorbike"));
        assert_eq!(vehicle_label(5), Some("bus"));
        assert_eq!(vehicle_label(7), Some("truck"));
        assert_eq!(vehicle_label(0), None);
    }

    /// Exact DeepStream nvinfer property keys — DeepStream silently ignores
    /// unknown keys, so a typo here would be invisible at runtime.
    #[test]
    fn yolo_properties_include_required_keys() {
        let engine = PathBuf::from("/tmp/yolo11n.onnx_b1_gpu0_fp16.engine");
        let props = yolo11n_properties(Path::new("/tmp/yolo11n.onnx"), 0, &engine);
        for key in [
            "onnx-file",
            "model-engine-file",
            "gpu-id",
            "network-mode",
            "batch-size",
            "net-scale-factor",
            "output-blob-names",
        ] {
            assert!(props.contains_key(key), "missing nvinfer property: {key}");
        }
        assert_eq!(props.get("batch-size"), Some(&"1".to_string()));
        assert_eq!(props.get("network-mode"), Some(&"2".to_string()));
    }
}
