# Encoder Enums Reference

## Platform
| Variant | Usage |
|---|---|
| `Platform.DGPU` | Desktop GPU (x86_64) |
| `Platform.JETSON` | NVIDIA Jetson |
| `Platform.from_name("dgpu")` | ⚠ ValueError on unknown |

## RateControl
| Variant |
|---|
| `RateControl.VARIABLE_BITRATE` |
| `RateControl.CONSTANT_BITRATE` |
| `RateControl.CONSTANT_QP` |
| `RateControl.from_name(name)` |

## H264Profile
`BASELINE`, `MAIN`, `HIGH`, `HIGH444`

## HevcProfile
`MAIN`, `MAIN10`, `FREXT`

## DgpuPreset
`P1` through `P7` (P1=fastest, P7=best quality)

## TuningPreset
`HIGH_QUALITY`, `LOW_LATENCY`, `ULTRA_LOW_LATENCY`, `LOSSLESS`

## JetsonPresetLevel
`DISABLED`, `ULTRA_FAST`, `FAST`, `MEDIUM`, `SLOW`

## Codec (from savant_rs.gstreamer)
`H264`, `HEVC`, `JPEG`, `AV1`, `PNG`, `RAW_RGBA`, `RAW_RGB`

## VideoFormat (from savant_rs.deepstream)
`RGBA`, `BGRx`, `NV12`, `NV21`, `I420`, `UYVY`, `GRAY8`

## MemType (from savant_rs.deepstream)
`DEFAULT`, `CUDA_PINNED`, `CUDA_DEVICE`, `CUDA_UNIFIED`, `SURFACE_ARRAY`, `HANDLE`, `SYSTEM`

## Padding (from savant_rs.deepstream)
`NONE`, `RIGHT_BOTTOM`, `SYMMETRIC` (DEF)

## Interpolation (from savant_rs.deepstream)
`NEAREST`, `BILINEAR` (DEF), `GPU_CUBIC_VIC_5TAP`, `GPU_SUPER_VIC_10TAP`, `GPU_LANCZOS_VIC_SMART`, `GPU_IGNORED_VIC_NICEST`, `DEFAULT`

## ComputeMode (from savant_rs.deepstream)
`DEFAULT`, `GPU`, `VIC`

---

## PtsResetPolicy (from savant_rs.picasso)
Policy for handling non-monotonic (decreasing) PTS values.

| Factory Static | Meaning |
|---|---|
| `PtsResetPolicy.eos_on_decreasing_pts()` | Emit synthetic EOS before recreating the encoder (default) |
| `PtsResetPolicy.recreate_on_decreasing_pts()` | Silently recreate the encoder without emitting EOS |

## StreamResetReason (from savant_rs.picasso)
Reason the worker's encoder was reset. Passed to the `on_stream_reset` callback.

| Property | Type | Notes |
|---|---|---|
| `last_pts_ns` | int | PTS of the last successfully accepted frame (nanoseconds) |
| `new_pts_ns` | int | PTS of the incoming frame that triggered the reset (nanoseconds) |

## CallbackInvocationOrder (from savant_rs.picasso)
Controls when the `on_gpumat` callback fires relative to Skia rendering.

| Variant | Meaning |
|---|---|
| `CallbackInvocationOrder.SkiaGpuMat` | Skia render then `on_gpumat` (default) |
| `CallbackInvocationOrder.GpuMatSkia` | `on_gpumat` then Skia render |
| `CallbackInvocationOrder.GpuMatSkiaGpuMat` | `on_gpumat` before **and** after Skia render |
| `CallbackInvocationOrder.from_name(name)` | ⚠ ValueError on unknown |

---

## Encoder Property Structs

All fields Optional, DEF: None.

### H264DgpuProps / HevcDgpuProps
`bitrate`, `control_rate`, `profile` (H264Profile/HevcProfile), `iframeinterval`, `idrinterval`, `preset` (DgpuPreset), `tuning_info` (TuningPreset), `qp_range` (str), `const_qp` (str), `init_qp` (str), `max_bitrate`, `vbv_buf_size`, `vbv_init`, `cq`, `aq`, `temporal_aq` (bool), `extended_colorformat` (bool)

### H264JetsonProps
`bitrate`, `control_rate`, `profile`, `iframeinterval`, `idrinterval`, `preset_level` (JetsonPresetLevel), `peak_bitrate`, `vbv_size`, `qp_range`, `quant_i_frames`, `quant_p_frames`, `ratecontrol_enable`, `maxperf_enable`, `two_pass_cbr`, `num_ref_frames`, `insert_sps_pps`, `insert_aud`, `insert_vui`, `disable_cabac`

### HevcJetsonProps
Same as H264JetsonProps except: no `insert_aud`, `insert_vui`, `disable_cabac`; adds `enable_lossless`

### JpegProps
`quality: Optional[int]`

### PngProps
`compression_level: Optional[int]` (0–9, default: 6)

### Av1DgpuProps
`bitrate`, `control_rate`, `iframeinterval`, `idrinterval`, `preset` (DgpuPreset), `tuning_info` (TuningPreset), `qp_range`, `max_bitrate`, `vbv_buf_size`, `vbv_init`, `cq`, `aq`, `temporal_aq`
