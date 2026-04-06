# DeepStream Encoders Architecture

## Module Tree
```
deepstream_encoders/src/
├── lib.rs            # EncoderConfig, EncodedFrame, re-exports
├── encoder.rs        # NvEncoder: GStreamer pipeline, submit/pull/finish
├── error.rs          # EncoderError enum
├── properties.rs     # EncoderProperties enum, per-codec/platform structs
└── prelude.rs        # Convenience re-exports
```

## GStreamer Pipeline Variants

### NVENC codecs (H264, HEVC, AV1) — NVENC required
```
appsrc (NVMM, native fmt) → encoder → parser → appsink
```
- No `nvvideoconvert` in the NVENC pipeline — format conversion (e.g. RGBA→NV12) is done **outside** the pipeline via `NvBufSurfTransform` with a dedicated non-blocking CUDA stream (see Format Conversion Architecture below)

### JPEG — nvjpegenc required
```
appsrc (NVMM, I420) → [nvvideoconvert*] → nvjpegenc → jpegparse → appsink
```
- On Jetson: `nvvideoconvert` with `disable-passthrough=true` is inserted before `nvjpegenc` to avoid "Surface not registered" hang

### PNG — CPU-based, always available
```
appsrc (NVMM, RGBA) → nvvideoconvert → pngenc → appsink
```
- `nvvideoconvert` handles NVMM→system-memory transfer
- Requires `VideoFormat::RGBA`

### Raw pseudoencoders (RawRgba, RawRgb) — always available
```
appsrc (NVMM) → nvvideoconvert → capsfilter(video/x-raw,format=RGBA|RGB) → appsink
```
- On Jetson (aarch64): `nvvideoconvert` gets `compute-hw=1` to bypass VIC limitations
- Output: tightly-packed pixel data (stride padding stripped by `extract_raw_pixels`)

### Raw pseudoencoder (RawNv12) — always available
```
appsrc (NVMM) → nvvideoconvert → capsfilter(video/x-raw,format=NV12) → appsink
```
- Output: tightly-packed NV12 pixel data (Y plane followed by interleaved UV plane, stride padding stripped)

## Format Conversion Architecture

When user format differs from encoder-native format:
```
User buffer (RGBA, NVMM)
  ↓ NvBufSurfTransform (non-blocking CUDA stream)
Native buffer (NV12/I420, NVMM)
  ↓ appsrc push
GStreamer pipeline
```

⚠ The NvBufSurfTransform uses a **dedicated non-blocking CUDA stream**, NOT the default stream (stream 0). This avoids the serialization bottleneck caused by `nvvideoconvert` using the legacy default stream, which blocks all GPU work across the process.

### ConvertContext (internal)
```rust
struct ConvertContext {
    native_generator: BufferGenerator,  // NV12/I420 pool
    cuda_stream: CudaStream,                         // non-blocking CUDA stream
}
```
Created when: `format != native_format` and codec is NOT PNG/Raw.
Destroyed on: `NvEncoder::drop()`.

## Native Formats per Codec
| Codec | Native Format |
|---|---|
| H264, HEVC, AV1 | NV12 |
| JPEG | I420 |
| PNG | RGBA |
| RawRgba | RGBA |
| RawRgb | RGB |
| RawNv12 | NV12 |

## NVENC Detection

At `NvEncoder::new()` time, if codec requires NVENC (H264/HEVC/AV1):
```rust
let has_nvenc = nvidia_gpu_utils::has_nvenc(config.gpu_id).unwrap_or(false);
if !has_nvenc {
    return Err(EncoderError::NvencNotAvailable { codec, gpu_id });
}
```

## B-Frame Enforcement

B-frames are **always** disabled:
1. The typed property API has no B-frame fields
2. `force_disable_b_frames()` iterates known B-frame property names and sets them to 0
3. Output PTS/DTS validation detects B-frame reordering at runtime

## Buffer Pool Size

On dGPU: `pool_size=1`. The NVENC hardware encoder may continue DMA-reading from GPU memory after GStreamer releases the buffer reference. A pool of 1 forces serialization.

On Jetson (aarch64) for H264/HEVC/AV1: `pool_size=4`. On Jetson for other codecs: `pool_size=1`.

## PTS Tracking

- `last_input_pts_ns`: validates strictly monotonic input PTS
- `last_output_pts_ns`: detects B-frame reordering in output
- `pts_map: HashMap<u64, (u128, Option<u64>)>`: maps output PTS → (frame_id, duration)
- Codec header buffers (e.g. AV1 sequence header) are not in pts_map; validation skipped for them

## Raw Pixel Extraction

`extract_raw_pixels` uses `gst_video::VideoInfo` from sample caps to:
1. Get width, height, stride, bytes-per-pixel
2. Map buffer readable
3. If stride == row_bytes: return slice directly
4. If stride != row_bytes: copy row-by-row, stripping padding

This produces tightly-packed pixel data regardless of GPU stride alignment.
