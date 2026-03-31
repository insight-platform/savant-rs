# Decoder MP4 Test Assets

This directory contains generated MP4 assets for decoder E2E tests.

## Coverage

- Codecs: H.264, HEVC, VP8, VP9, AV1, JPEG
- Colorspaces:
  - BT.709 (8-bit)
  - BT.2020 (10-bit, where applicable)
- B-frame variants (where applicable):
  - H.264: `ip` (no B-frames), `ipb` (B-frames enabled)
  - HEVC: `ip` (no B-frames), `ipb` (B-frames enabled)

## Files

- `test_h264_bt709_ip.mp4`
- `test_h264_bt709_ipb.mp4`
- `test_hevc_bt709_ip.mp4`
- `test_hevc_bt709_ipb.mp4`
- `test_vp8_bt709.mp4`
- `test_vp9_bt709.mp4`
- `test_av1_bt709.mp4`
- `test_jpeg_bt709.mp4`
- `test_h264_bt2020_ip.mp4`
- `test_h264_bt2020_ipb.mp4`
- `test_hevc_bt2020_ip.mp4`
- `test_hevc_bt2020_ipb.mp4`
- `test_vp9_bt2020.mp4`

## Metadata

Per-file device support metadata and generation pipelines are stored in:

- `manifest.json`
