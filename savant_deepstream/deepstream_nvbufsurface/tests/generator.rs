//! Integration tests for [`NvBufSurfaceGenerator`] constructors, caps,
//! surface acquisition, and memory type handling.

mod common;

use deepstream_nvbufsurface::{
    NvBufSurfaceGenerator, NvBufSurfaceMemType, SavantIdMeta, SavantIdMetaKind,
};
use gstreamer as gst;

// ─── Constructor tests ───────────────────────────────────────────────────────

#[test]
fn test_create_generator() {
    common::init();

    let generator = NvBufSurfaceGenerator::new(
        "RGBA", 640, 480, 30, 1,
        0, NvBufSurfaceMemType::Default,
    )
    .expect("Failed to create NvBufSurfaceGenerator");

    // Generator should be valid and pool should be active
    drop(generator);
}

#[test]
fn test_from_caps() {
    common::init();

    let caps = gst::Caps::builder("video/x-raw")
        .field("format", "RGBA")
        .field("width", 640i32)
        .field("height", 480i32)
        .field("framerate", gst::Fraction::new(30, 1))
        .build();

    let generator =
        NvBufSurfaceGenerator::from_caps(&caps, 0, NvBufSurfaceMemType::Default)
            .expect("Failed to create via from_caps");

    let buffer = generator.acquire_surface(None).unwrap();
    assert!(buffer.n_memory() > 0);
}

#[test]
fn test_builder_defaults() {
    common::init();

    let generator = NvBufSurfaceGenerator::builder("RGBA", 640, 480)
        .build()
        .expect("Failed to create via builder with defaults");

    let buffer = generator.acquire_surface(None).unwrap();
    assert!(buffer.n_memory() > 0);
}

#[test]
fn test_builder_full() {
    common::init();

    let generator = NvBufSurfaceGenerator::builder("NV12", 1920, 1080)
        .fps(60, 1)
        .gpu_id(0)
        .mem_type(NvBufSurfaceMemType::Default)
        .min_buffers(2)
        .max_buffers(8)
        .build()
        .expect("Failed to create via builder with all options");

    let buffer = generator.acquire_surface(None).unwrap();
    assert!(buffer.n_memory() > 0);
}

// ─── Surface acquisition tests ───────────────────────────────────────────────

#[test]
fn test_acquire_surface() {
    common::init();

    let generator = NvBufSurfaceGenerator::new(
        "RGBA", 640, 480, 30, 1,
        0, NvBufSurfaceMemType::Default,
    )
    .unwrap();

    let buffer = generator.acquire_surface(None).unwrap();
    assert!(buffer.n_memory() > 0, "Buffer should have memory");
}

#[test]
fn test_acquire_surface_with_id() {
    common::init();

    let generator = NvBufSurfaceGenerator::new(
        "RGBA", 640, 480, 30, 1,
        0, NvBufSurfaceMemType::Default,
    )
    .unwrap();

    let buffer = generator.acquire_surface(Some(42)).unwrap();
    assert!(buffer.n_memory() > 0, "Buffer should have memory");

    // Verify the SavantIdMeta was attached
    let meta = buffer
        .meta::<SavantIdMeta>()
        .expect("Buffer should have SavantIdMeta");
    assert_eq!(meta.ids(), &[SavantIdMetaKind::Frame(42)]);
}

#[test]
fn test_acquire_surface_with_ptr_and_id() {
    common::init();

    let generator = NvBufSurfaceGenerator::new(
        "RGBA", 640, 480, 30, 1,
        0, NvBufSurfaceMemType::Default,
    )
    .unwrap();

    let (buffer, _data_ptr, _pitch) = generator.acquire_surface_with_ptr(Some(99)).unwrap();
    assert!(buffer.n_memory() > 0);

    let meta = buffer
        .meta::<SavantIdMeta>()
        .expect("Buffer should have SavantIdMeta");
    assert_eq!(meta.ids(), &[SavantIdMetaKind::Frame(99)]);
}

#[test]
fn test_acquire_surface_without_id_has_no_meta() {
    common::init();

    let generator = NvBufSurfaceGenerator::new(
        "RGBA", 640, 480, 30, 1,
        0, NvBufSurfaceMemType::Default,
    )
    .unwrap();

    let buffer = generator.acquire_surface(None).unwrap();
    assert!(
        buffer.meta::<SavantIdMeta>().is_none(),
        "Buffer without id should not have SavantIdMeta"
    );
}

#[test]
fn test_acquire_multiple_surfaces() {
    common::init();

    let generator = NvBufSurfaceGenerator::new(
        "RGBA", 320, 240, 30, 1,
        0, NvBufSurfaceMemType::Default,
    )
    .unwrap();

    for i in 0..10 {
        let buffer = generator
            .acquire_surface(Some(i as i64))
            .unwrap_or_else(|_| panic!("Failed to acquire surface {}", i));
        assert!(buffer.n_memory() > 0, "Buffer {} should have memory", i);

        let meta = buffer
            .meta::<SavantIdMeta>()
            .unwrap_or_else(|| panic!("Buffer {} should have SavantIdMeta", i));
        assert_eq!(meta.ids(), &[SavantIdMetaKind::Frame(i as i64)]);
    }
}

// ─── create_surface tests ────────────────────────────────────────────────────

#[test]
fn test_create_surface() {
    common::init();

    let generator = NvBufSurfaceGenerator::new(
        "RGBA", 640, 480, 30, 1,
        0, NvBufSurfaceMemType::Default,
    )
    .unwrap();

    let mut buffer = gst::Buffer::new();
    generator
        .create_surface(buffer.make_mut(), None)
        .expect("Failed to create surface");

    assert!(
        buffer.n_memory() > 0,
        "Buffer should have memory after create_surface"
    );
}

#[test]
fn test_create_surface_with_id() {
    common::init();

    let generator = NvBufSurfaceGenerator::new(
        "RGBA", 640, 480, 30, 1,
        0, NvBufSurfaceMemType::Default,
    )
    .unwrap();

    let mut buffer = gst::Buffer::new();
    generator
        .create_surface(buffer.make_mut(), Some(7))
        .expect("Failed to create surface with id");

    assert!(buffer.n_memory() > 0);

    let meta = buffer
        .meta::<SavantIdMeta>()
        .expect("Buffer should have SavantIdMeta after create_surface with id");
    assert_eq!(meta.ids(), &[SavantIdMetaKind::Frame(7)]);
}

#[test]
fn test_create_surface_raw() {
    common::init();

    let generator = NvBufSurfaceGenerator::new(
        "RGBA", 640, 480, 30, 1,
        0, NvBufSurfaceMemType::Default,
    )
    .unwrap();

    let mut buffer = gst::Buffer::new();
    let raw_ptr = buffer.make_mut().as_mut_ptr();
    unsafe {
        generator
            .create_surface_raw(raw_ptr, None)
            .expect("Failed to create surface via raw API");
    }

    assert!(
        buffer.n_memory() > 0,
        "Buffer should have memory after create_surface_raw"
    );
}

#[test]
fn test_null_pointer_error() {
    common::init();

    let generator = NvBufSurfaceGenerator::new(
        "RGBA", 640, 480, 30, 1,
        0, NvBufSurfaceMemType::Default,
    )
    .unwrap();

    let result = unsafe { generator.create_surface_raw(std::ptr::null_mut(), None) };
    assert!(result.is_err(), "Should fail with null pointer");
}

// ─── Caps tests ──────────────────────────────────────────────────────────────

#[test]
fn test_nvmm_caps() {
    common::init();

    let generator = NvBufSurfaceGenerator::new(
        "NV12", 640, 480, 30, 1,
        0, NvBufSurfaceMemType::Default,
    )
    .unwrap();

    let caps = generator.nvmm_caps();
    let caps_str = caps.to_string();

    assert!(
        caps_str.contains("memory:NVMM"),
        "Caps should have NVMM feature: {}",
        caps_str
    );
    assert!(
        caps_str.contains("NV12"),
        "Caps should have NV12 format: {}",
        caps_str
    );
    assert!(
        caps_str.contains("640"),
        "Caps should have width 640: {}",
        caps_str
    );
    assert!(
        caps_str.contains("480"),
        "Caps should have height 480: {}",
        caps_str
    );
}

#[test]
fn test_raw_caps() {
    common::init();

    let generator = NvBufSurfaceGenerator::new(
        "RGBA", 1920, 1080, 60, 1,
        0, NvBufSurfaceMemType::Default,
    )
    .unwrap();

    let caps = generator.raw_caps();
    let caps_str = caps.to_string();

    assert!(
        !caps_str.contains("memory:NVMM"),
        "Raw caps should NOT have NVMM feature: {}",
        caps_str
    );
    assert!(
        caps_str.contains("RGBA"),
        "Caps should have RGBA format: {}",
        caps_str
    );
    assert!(
        caps_str.contains("1920"),
        "Caps should have width 1920: {}",
        caps_str
    );
    assert!(
        caps_str.contains("1080"),
        "Caps should have height 1080: {}",
        caps_str
    );
}

// ─── Different resolutions / mem types ───────────────────────────────────────

#[test]
fn test_different_resolutions() {
    common::init();

    for (w, h) in [(320, 240), (640, 480), (1280, 720), (1920, 1080)] {
        let generator = NvBufSurfaceGenerator::new(
            "RGBA", w, h, 30, 1,
            0, NvBufSurfaceMemType::Default,
        )
        .unwrap_or_else(|_| panic!("Failed to create generator for {}x{}", w, h));

        let buffer = generator
            .acquire_surface(None)
            .unwrap_or_else(|_| panic!("Failed to acquire surface for {}x{}", w, h));
        assert!(
            buffer.n_memory() > 0,
            "Buffer for {}x{} should have memory",
            w,
            h
        );
    }
}

#[test]
fn test_cuda_device_mem_type() {
    common::init();

    let generator = NvBufSurfaceGenerator::new(
        "RGBA", 640, 480, 30, 1,
        0, NvBufSurfaceMemType::CudaDevice,
    )
    .expect("Failed to create NvBufSurfaceGenerator with CudaDevice mem type");

    let buffer = generator
        .acquire_surface(None)
        .expect("Failed to acquire surface with CudaDevice mem type");
    assert!(buffer.n_memory() > 0);
}

#[test]
fn test_mem_type_conversions() {
    assert_eq!(NvBufSurfaceMemType::from(0), NvBufSurfaceMemType::Default);
    assert_eq!(
        NvBufSurfaceMemType::from(1),
        NvBufSurfaceMemType::CudaPinned
    );
    assert_eq!(
        NvBufSurfaceMemType::from(2),
        NvBufSurfaceMemType::CudaDevice
    );
    assert_eq!(
        NvBufSurfaceMemType::from(3),
        NvBufSurfaceMemType::CudaUnified
    );
    assert_eq!(
        NvBufSurfaceMemType::from(4),
        NvBufSurfaceMemType::SurfaceArray
    );
    assert_eq!(NvBufSurfaceMemType::from(5), NvBufSurfaceMemType::Handle);
    assert_eq!(NvBufSurfaceMemType::from(6), NvBufSurfaceMemType::System);
    assert_eq!(NvBufSurfaceMemType::from(99), NvBufSurfaceMemType::Default);

    assert_eq!(u32::from(NvBufSurfaceMemType::Default), 0);
    assert_eq!(u32::from(NvBufSurfaceMemType::CudaDevice), 2);
}
