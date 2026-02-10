//! Integration tests for NvBufSurfTransform via the high-level
//! `NvBufSurfaceGenerator::transform()` API.

mod common;

use deepstream_nvbufsurface::{
    ComputeMode, Interpolation, NvBufSurfaceGenerator, NvBufSurfaceMemType, Padding, Rect,
    SavantIdMeta, TransformConfig,
};

/// Helper: create a generator with the given format and dimensions.
fn make_gen(format: &str, w: u32, h: u32) -> NvBufSurfaceGenerator {
    NvBufSurfaceGenerator::builder(format, w, h)
        .gpu_id(0)
        .mem_type(NvBufSurfaceMemType::Default)
        .min_buffers(2)
        .max_buffers(2)
        .build()
        .expect("failed to build generator")
}

// ─── Basic Transform Tests ──────────────────────────────────────────────────

#[test]
fn transform_rgba_to_rgba_same_size() {
    common::init();

    let src_gen = make_gen("RGBA", 640, 480);
    let dst_gen = make_gen("RGBA", 640, 480);

    let src_buf = src_gen.acquire_surface(None).unwrap();
    let config = TransformConfig::default();
    let dst_buf = dst_gen.transform(&src_buf, &config, None).unwrap();
    assert!(dst_buf.size() > 0);
}

#[test]
fn transform_rgba_downscale_symmetric() {
    common::init();

    let src_gen = make_gen("RGBA", 1920, 1080);
    let dst_gen = make_gen("RGBA", 640, 640);

    let src_buf = src_gen.acquire_surface(None).unwrap();
    let config = TransformConfig {
        padding: Padding::Symmetric,
        interpolation: Interpolation::Bilinear,
        ..Default::default()
    };
    let dst_buf = dst_gen.transform(&src_buf, &config, None).unwrap();
    assert!(dst_buf.size() > 0);
}

#[test]
fn transform_rgba_downscale_right_bottom() {
    common::init();

    let src_gen = make_gen("RGBA", 1920, 1080);
    let dst_gen = make_gen("RGBA", 640, 640);

    let src_buf = src_gen.acquire_surface(None).unwrap();
    let config = TransformConfig {
        padding: Padding::RightBottom,
        interpolation: Interpolation::Bilinear,
        ..Default::default()
    };
    let dst_buf = dst_gen.transform(&src_buf, &config, None).unwrap();
    assert!(dst_buf.size() > 0);
}

#[test]
fn transform_rgba_no_padding() {
    common::init();

    let src_gen = make_gen("RGBA", 1920, 1080);
    let dst_gen = make_gen("RGBA", 640, 480);

    let src_buf = src_gen.acquire_surface(None).unwrap();
    let config = TransformConfig {
        padding: Padding::None,
        interpolation: Interpolation::Nearest,
        ..Default::default()
    };
    let dst_buf = dst_gen.transform(&src_buf, &config, None).unwrap();
    assert!(dst_buf.size() > 0);
}

#[test]
fn transform_nv12_to_rgba() {
    common::init();

    let src_gen = make_gen("NV12", 1920, 1080);
    let dst_gen = make_gen("RGBA", 640, 480);

    let src_buf = src_gen.acquire_surface(None).unwrap();
    let config = TransformConfig {
        padding: Padding::Symmetric,
        interpolation: Interpolation::Bilinear,
        ..Default::default()
    };
    let dst_buf = dst_gen.transform(&src_buf, &config, None).unwrap();
    assert!(dst_buf.size() > 0);
}

#[test]
fn transform_rgba_to_nv12() {
    common::init();

    let src_gen = make_gen("RGBA", 1920, 1080);
    let dst_gen = make_gen("NV12", 640, 480);

    let src_buf = src_gen.acquire_surface(None).unwrap();
    let config = TransformConfig {
        padding: Padding::Symmetric,
        interpolation: Interpolation::Bilinear,
        ..Default::default()
    };
    let dst_buf = dst_gen.transform(&src_buf, &config, None).unwrap();
    assert!(dst_buf.size() > 0);
}

// ─── Interpolation Methods ──────────────────────────────────────────────────

#[test]
fn transform_all_interpolation_methods() {
    common::init();

    let src_gen = make_gen("RGBA", 1920, 1080);
    let dst_gen = make_gen("RGBA", 640, 480);

    let methods = [
        Interpolation::Nearest,
        Interpolation::Bilinear,
        Interpolation::Algo1,
        Interpolation::Algo2,
        Interpolation::Algo3,
        Interpolation::Default,
    ];

    for interp in methods {
        let src_buf = src_gen.acquire_surface(None).unwrap();
        let config = TransformConfig {
            padding: Padding::None,
            interpolation: interp,
            ..Default::default()
        };
        let dst_buf = dst_gen.transform(&src_buf, &config, None).unwrap();
        assert!(dst_buf.size() > 0, "interpolation {:?} failed", interp);
    }
}

// ─── Source Crop ─────────────────────────────────────────────────────────────

#[test]
fn transform_with_src_crop() {
    common::init();

    let src_gen = make_gen("RGBA", 1920, 1080);
    let dst_gen = make_gen("RGBA", 640, 480);

    let src_buf = src_gen.acquire_surface(None).unwrap();
    let config = TransformConfig {
        padding: Padding::Symmetric,
        interpolation: Interpolation::Bilinear,
        src_rect: Some(Rect {
            top: 100,
            left: 200,
            width: 800,
            height: 600,
        }),
        compute_mode: ComputeMode::Default,
    };
    let dst_buf = dst_gen.transform(&src_buf, &config, None).unwrap();
    assert!(dst_buf.size() > 0);
}

// ─── SavantIdMeta Preservation ──────────────────────────────────────────────

#[test]
fn transform_preserves_savant_id_meta() {
    common::init();

    let src_gen = make_gen("RGBA", 1920, 1080);
    let dst_gen = make_gen("RGBA", 640, 640);

    // Acquire source with ID
    let src_buf = src_gen.acquire_surface(Some(42)).unwrap();
    assert!(
        src_buf.meta::<SavantIdMeta>().is_some(),
        "source should have SavantIdMeta"
    );

    // Transform with a new ID (the transform creates a new destination buffer
    // with its own ID)
    let config = TransformConfig::default();
    let dst_buf = dst_gen.transform(&src_buf, &config, Some(99)).unwrap();
    let meta = dst_buf
        .meta::<SavantIdMeta>()
        .expect("destination should have SavantIdMeta");
    let ids = meta.ids();
    assert_eq!(ids.len(), 1);
    match &ids[0] {
        deepstream_nvbufsurface::SavantIdMetaKind::Frame(id) => assert_eq!(*id, 99),
        other => panic!("expected Frame(99), got {:?}", other),
    }
}

// ─── transform_with_ptr ─────────────────────────────────────────────────────

#[test]
fn transform_with_ptr_returns_valid_data() {
    common::init();

    let src_gen = make_gen("RGBA", 1920, 1080);
    let dst_gen = make_gen("RGBA", 640, 640);

    let src_buf = src_gen.acquire_surface(None).unwrap();
    let config = TransformConfig {
        padding: Padding::Symmetric,
        interpolation: Interpolation::Bilinear,
        ..Default::default()
    };
    let (dst_buf, data_ptr, pitch) = dst_gen
        .transform_with_ptr(&src_buf, &config, None)
        .unwrap();
    assert!(dst_buf.size() > 0);
    assert!(!data_ptr.is_null(), "data_ptr should not be null");
    assert!(pitch > 0, "pitch should be > 0");
}

// ─── Compute Mode (GPU) ────────────────────────────────────────────────────

#[test]
fn transform_gpu_compute_mode() {
    common::init();

    let src_gen = make_gen("RGBA", 1920, 1080);
    let dst_gen = make_gen("RGBA", 640, 480);

    let src_buf = src_gen.acquire_surface(None).unwrap();
    let config = TransformConfig {
        padding: Padding::Symmetric,
        interpolation: Interpolation::Bilinear,
        compute_mode: ComputeMode::Gpu,
        ..Default::default()
    };
    let dst_buf = dst_gen.transform(&src_buf, &config, None).unwrap();
    assert!(dst_buf.size() > 0);
}

// ─── Upscale ────────────────────────────────────────────────────────────────

#[test]
fn transform_upscale_symmetric() {
    common::init();

    let src_gen = make_gen("RGBA", 320, 240);
    let dst_gen = make_gen("RGBA", 1920, 1080);

    let src_buf = src_gen.acquire_surface(None).unwrap();
    let config = TransformConfig {
        padding: Padding::Symmetric,
        interpolation: Interpolation::Bilinear,
        ..Default::default()
    };
    let dst_buf = dst_gen.transform(&src_buf, &config, None).unwrap();
    assert!(dst_buf.size() > 0);
}
