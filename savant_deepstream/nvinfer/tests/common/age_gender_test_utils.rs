//! Shared helpers for age-gender–style integration tests.
//!
//! Used by `test_age_gender`, `test_roi_crop_fidelity`, and `test_oversized_batch`.
#![allow(dead_code)]
// This module is declared from `common/mod.rs`, which every integration test
// crate pulls in; only some binaries `use` these items, so `dead_code` is
// expected on the others.

use candle_core::{DType, Device, Tensor};
use nvinfer::{DataType, TensorView};
use rand::rngs::SmallRng;
use rand::Rng;
use std::path::Path;

/// `gstnvinfer.cpp` applies `GST_ROUND_UP_2` to crop coordinates; even-aligned
/// placements avoid misaligned crops.
pub const PLACEMENT_ALIGN: u32 = 2;

/// Place `count` non-overlapping `w x h` rectangles on a `fw x fh` canvas.
/// Returns `(left, top)` for each placement. Panics if placement fails.
pub fn place_non_overlapping(
    rng: &mut SmallRng,
    fw: u32,
    fh: u32,
    w: u32,
    h: u32,
    count: usize,
) -> Vec<(u32, u32)> {
    let mut placed: Vec<(u32, u32)> = Vec::with_capacity(count);
    let max_x = (fw - w) / PLACEMENT_ALIGN;
    let max_y = (fh - h) / PLACEMENT_ALIGN;
    for _ in 0..count {
        for attempt in 0..10_000 {
            let x = rng.random_range(0..=max_x) * PLACEMENT_ALIGN;
            let y = rng.random_range(0..=max_y) * PLACEMENT_ALIGN;
            let overlaps = placed
                .iter()
                .any(|&(px, py)| x < px + w && x + w > px && y < py + h && y + h > py);
            if !overlaps {
                placed.push((x, y));
                break;
            }
            assert!(attempt < 9_999, "failed to place image without overlap");
        }
    }
    placed
}

/// Load all face JPEGs from `dir`, sorted by filename.
/// Each image must be exactly `face_w` × `face_h` RGBA after decode.
pub fn load_face_images(dir: &Path, face_w: u32, face_h: u32) -> Vec<(String, Vec<u8>)> {
    let mut images: Vec<(String, Vec<u8>)> = Vec::new();
    for entry in std::fs::read_dir(dir).expect("read face image dir") {
        let entry = entry.unwrap();
        let fname = entry.file_name().to_string_lossy().to_string();
        if !fname.ends_with(".jpg") {
            continue;
        }
        let img = image::open(entry.path())
            .unwrap_or_else(|e| panic!("failed to open {fname}: {e}"))
            .to_rgba8();
        assert_eq!(img.width(), face_w, "{fname}: unexpected width");
        assert_eq!(img.height(), face_h, "{fname}: unexpected height");
        images.push((fname, img.into_raw()));
    }
    images.sort_by(|a, b| a.0.cmp(&b.0));
    images
}

/// Build a candle tensor from a nvinfer [`TensorView`], handling fp16 and fp32.
pub fn to_candle_tensor(tv: &TensorView, shape: &[usize]) -> candle_core::Result<Tensor> {
    match tv.data_type {
        DataType::Half => {
            let raw: &[half::f16] = unsafe { tv.as_slice() };
            Tensor::from_slice(raw, shape, &Device::Cpu)?.to_dtype(DType::F32)
        }
        DataType::Float => {
            let raw: &[f32] = unsafe { tv.as_slice() };
            Tensor::from_slice(raw, shape, &Device::Cpu)
        }
        other => panic!("unsupported tensor dtype: {other:?}"),
    }
}

/// Decode age: weighted sum of 101 class probabilities.
pub fn decode_age(tensor: &TensorView) -> candle_core::Result<f32> {
    let probs = to_candle_tensor(tensor, &[101])?;
    let age_range = Tensor::arange(0f32, 101f32, &Device::Cpu)?;
    probs.mul(&age_range)?.sum_all()?.to_scalar::<f32>()
}

/// Decode gender: argmax over [male, female] logits.
pub fn decode_gender(tensor: &TensorView) -> candle_core::Result<String> {
    let t = to_candle_tensor(tensor, &[2])?;
    let idx = t.argmax(0)?.to_scalar::<u32>()?;
    Ok(if idx == 0 { "male" } else { "female" }.into())
}
