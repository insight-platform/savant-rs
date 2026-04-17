# NVIDIA NVENC / NVDEC Support Matrix — Extract

**Source:** <https://developer.nvidia.com/video-encode-decode-support-matrix>
(extract captured 2026-04-17; always re-check the live page for the latest
SKUs — it is NVIDIA's single source of truth and is updated with each new GPU
generation).

This document is a condensed, savant-rs–oriented extract of NVIDIA's
Video Encode & Decode Support Matrix. It is intentionally **per-generation**
(not per-SKU) because for our purposes every SKU in a given NVDEC/NVENC
generation exposes the same codec/bit-depth/chroma capability set — the
only per-SKU differences are NVENC/NVDEC unit count, concurrent session
limit, Desktop vs. Mobile, and AV1 *encode* fuse bits on early Ada SKUs.

See also:

* [`kb/decoders/caveats.md`](./decoders/caveats.md) — runtime caveats,
  error-801 gotcha, and how the matrix is encoded into the e2e test
  manifest.
* [`savant_deepstream/decoders/assets/manifest.json`](../savant_deepstream/decoders/assets/manifest.json)
  — `supported_platforms` arrays that encode the matrix per test asset.
* `savant_deepstream/nvidia_gpu_utils/` — `is_jetson_kernel()` and related
  runtime platform detection, used to gate platform-specific pipeline
  construction.

---

## 1) Platform naming used in savant-rs

The e2e manifest uses these short names; they map to NVDEC/NVENC
generations as follows:

| `manifest.json` platform | GPU family          | Example SKUs                         | NVDEC gen       | NVENC gen       |
|--------------------------|---------------------|--------------------------------------|-----------------|-----------------|
| `turing`                 | Turing              | Tesla T4, RTX 2060/2070/2080, GTX 16 | 4th             | 7th             |
| `ampere`                 | Ampere              | RTX 30-series, A10, A16, A40         | 5th             | 7th             |
| `ada`                    | Ada Lovelace        | RTX 40-series, L4, L40, L40S         | 5th             | 8th             |
| `blackwell`              | Blackwell (dGPU)    | RTX 50-series, RTX PRO Blackwell     | 6th             | 9th             |
| `jetson_orin`            | Jetson Orin (Tegra) | Orin AGX / NX / Nano                 | Tegra (Orin)    | Tegra (Orin)    |

Notes:

* **Hopper** (H100/H200, GH200) has **no NVDEC and no NVENC** — the matrix
  lists them as `N/A` with 0 engines. They can only do JPEG decode via the
  NVJPG engine and software encode/decode via CUDA. Treat Hopper SKUs the
  same as "CPU-only" for savant-rs decode/encode purposes.
* **Blackwell data-center parts** (HGX B200/B300, GB200/GB300) also have
  **no NVDEC/NVENC**. Only the consumer/workstation Blackwell parts
  (RTX 50-series, RTX PRO Blackwell, DGX Spark/Station GB300) expose
  6th-gen NVDEC / 9th-gen NVENC.
* **Jetson Thor** (T4000/T5000, IGX T7000/T5000) is the successor to Orin
  and is listed in the matrix as NVDEC 6.1 gen / NVENC 9.1 gen —
  essentially Blackwell-class. When/if we add a `jetson_thor` platform to
  the manifest, use the Blackwell row of the decode/encode tables below.

---

## 2) NVDEC — Decode capability by generation

Columns mirror the matrix's NVDEC table (checkmarks collapsed to
YES/NO). Rows cover only the generations targeted by savant-rs.

| Gen (family)                            | MPEG-1/2/4 | VC-1 | VP8 | VP9 8b | VP9 10b | VP9 12b | H.264 4:2:0 8b | H.264 4:2:0 10b | H.264 4:2:2 (any) | HEVC 4:2:0 8/10/12b | HEVC 4:2:2 8/10/12b | HEVC 4:4:4 8/10/12b | AV1 8b | AV1 10b |
|-----------------------------------------|:----------:|:----:|:---:|:------:|:-------:|:-------:|:--------------:|:---------------:|:-----------------:|:-------------------:|:-------------------:|:-------------------:|:------:|:-------:|
| 4th gen — **Turing**                    |    YES     | YES  | YES |  YES   |   YES   |   YES   |      YES       |    **NO**       |      **NO**       |         YES         |        **NO**       |         YES         |  NO    |   NO    |
| 5th gen — **Ampere** (RTX 30, Ax)       |    YES     | YES  | YES |  YES   |   YES   |   YES   |      YES       |    **NO**       |      **NO**       |         YES         |        **NO**       |         YES         |  NO    |   NO    |
| 5th gen — **Ada Lovelace** (RTX 40, Lx) |    YES     | YES  | YES |  YES   |   YES   |   YES   |      YES       |    **NO**       |      **NO**       |         YES         |        **NO**       |         YES         |  YES   |   YES   |
| 6th gen — **Blackwell** (RTX 50, RTX PRO Blackwell) | YES | YES | YES | YES | YES | YES | YES | **YES** | **YES** | YES | **YES** | YES | YES | YES |
| Tegra — **Jetson Orin**                 |    YES     | YES  | YES |  YES   |   YES   |   NO    |      YES       |    **YES**      |      partial*     |         YES         |        partial*     |         YES         |  YES   |   YES   |

\* Jetson Orin 4:2:2 / 4:4:4 coverage varies by module revision and
JetPack version; the public Tegra Multimedia API docs are the
authoritative source. For the profiles savant-rs cares about (8-bit
4:2:0, 10-bit 4:2:0) Orin is fully capable.

### Key decode takeaways for savant-rs

1. **H.264 4:2:0 10-bit (High 10)** is the single most important split:
   * **NO** on Turing, Ampere, and Ada dGPU NVDEC.
   * **YES** on Blackwell dGPU NVDEC and Jetson Orin Tegra NVDEC.
   * Symptom on the "NO" platforms: `nvv4l2decoder` emits
     `NvV4l2VideoDec: Feature not supported on this GPU (Error Code: 801)`,
     `Failed to process frame`, then `streaming stopped, reason error (-5)`
     and zero frames decoded. See `kb/decoders/caveats.md §11`.
   * Manifest encoding: `test_h264_bt2020_ip.mp4` and
     `test_h264_bt2020_ipb.mp4` list `supported_platforms: ["blackwell",
     "jetson_orin"]`.
2. **H.264 4:2:2** (any bit depth) is only decoded by Blackwell. Not
   currently exercised by any test asset — if added, gate the manifest
   entry to `["blackwell", "jetson_orin"]` (Orin pending datasheet
   confirmation).
3. **HEVC Main10 (4:2:0 10-bit)** is universally supported on all
   generations we target (Turing and later). Safe default for 10-bit
   test assets.
4. **HEVC 4:2:2** (8/10/12-bit) is a Blackwell-only dGPU feature. Same
   gating rule as H.264 4:2:2.
5. **VP9** decode works on all dGPU generations (Turing+) and on Orin
   (8-bit / 10-bit, no 12-bit on Orin). VP9 **encode is not available
   on any NVIDIA NVENC** — there is no VP9 encoder in any NVENC
   generation.
6. **AV1 decode** starts at Ada (5th gen NVDEC on Ada silicon; Ampere
   5th gen is the same generation number but does *not* include AV1 —
   this is the exception that breaks the "gen-number = capabilities"
   rule; NVIDIA shipped AV1 decode only on Ada+ 5th-gen silicon).
7. **AV1 encode** starts at Ada 8th-gen NVENC (RTX 40-series / L4 /
   L40). Ampere 7th-gen NVENC does **not** do AV1 encode.

---

## 3) NVENC — Encode capability by generation

| Gen (family)                 | H.264 4:2:0 | H.264 4:2:2 | H.264 4:4:4 | H.264 Lossless | HEVC 4:2:0 4K | HEVC 4:2:2 | HEVC 4:4:4 4K | HEVC Lossless | HEVC 8K | HEVC 10-bit | HEVC B-frames | AV1 4:2:0 |
|------------------------------|:-----------:|:-----------:|:-----------:|:--------------:|:-------------:|:----------:|:-------------:|:-------------:|:-------:|:-----------:|:-------------:|:---------:|
| 7th gen — **Turing / Ampere**|     YES     |     NO      |     YES     |      YES       |      YES      |    NO      |     YES       |      YES      |   YES   |     YES     |      YES      |    NO     |
| 8th gen — **Ada Lovelace**   |     YES     |     NO      |     YES     |      YES       |      YES      |    NO      |     YES       |      YES      |   YES   |     YES     |      YES      |   YES     |
| 9th gen — **Blackwell**      |     YES     |    **YES**  |     YES     |      YES       |      YES      |  **YES**   |     YES       |      YES      |   YES   |     YES     |      YES      |   YES     |
| Tegra — **Jetson Orin**      |     YES     |     NO      |     YES     |      YES       |      YES      |    NO      |     YES       |      YES      |   YES   |     YES     |      YES      |    NO     |

### Key encode takeaways for savant-rs

1. **AV1 encode** requires Ada-class NVENC (8th gen) or newer. Not
   available on Turing, Ampere, or Jetson Orin.
2. **H.264 and HEVC 4:2:2 encode** is Blackwell-only (9th-gen NVENC).
3. **No VP9 encoder** exists in any NVIDIA hardware. Use
   `vp9enc`/`libvpx` (CPU) or `av1enc`/NVENC AV1 (on Ada+).
4. **Concurrent encoding sessions**: Consumer (GeForce) SKUs are limited
   to a few concurrent sessions by driver (typically 3–12 in recent
   drivers since the 2023 cap lift). Professional (RTX PRO / Quadro),
   Data Center (Lx/Ax), and Jetson SKUs are marked "Unrestricted". The
   matrix's *Max # of concurrent sessions* column is the authoritative
   number per SKU.
5. **# of NVENC / NVDEC engines per chip** matters for throughput. Most
   consumer SKUs have 1 NVENC / 1 NVDEC; RTX 5090 has 3 NVENC / 2 NVDEC;
   RTX PRO 6000 Blackwell Workstation has 4 NVENC / 4 NVDEC; L40/L40S
   has 3 NVENC / 3 NVDEC.

---

## 4) Why the matrix is "per-generation" here but not on NVIDIA's page

NVIDIA's page lists every SKU because marketing cares about per-SKU
numbers (engine counts, concurrent session caps, Desktop vs. Mobile,
consumer-vs-pro fuse bits). For **codec capability** — which is all
savant-rs cares about for pipeline construction and test gating — every
SKU inside a given NVDEC/NVENC *generation number* is identical, with
the following documented exceptions captured in the tables above:

* **Ampere 5th-gen NVDEC ≠ Ada 5th-gen NVDEC on AV1** — same generation
  number, different silicon, AV1 decode added on Ada only.
* **Blackwell data-center (HGX B200/B300, GB200/GB300)** has no NVENC /
  NVDEC at all, despite being "Blackwell". Only consumer/workstation
  Blackwell SKUs carry the 6th-gen NVDEC / 9th-gen NVENC.
* **Hopper** (H100/H200, GH200) has no NVENC / NVDEC at all.
* **GeForce MX450 / MX570 A** — Turing/Ampere mobile MX SKUs with the
  NVENC/NVDEC engines fused off; listed as `0 engines` in the matrix.

When in doubt, open the live matrix and Ctrl-F for the specific SKU.

---

## 5) How to keep this doc and the manifest in sync

When adding a new test asset that exercises a specific codec / bit-depth
/ chroma combination:

1. Look up the combination in §2 (decode) or §3 (encode).
2. Restrict `supported_platforms` in
   `savant_deepstream/decoders/assets/manifest.json` to only the
   generations that return YES.
3. Add an `unsupported_note` key with the rationale and a pointer back
   to this file and the source URL, so that the next maintainer
   doesn't over-eagerly re-expand the platform list.
4. If the combination crosses a known gotcha (H.264 10-bit, HEVC 4:2:2,
   VP9 on dGPU), add a bullet to `kb/decoders/caveats.md` too.

When a new GPU generation ships (e.g. post-Blackwell consumer Rubin /
whatever ships next):

1. Re-fetch the live matrix.
2. Add a new row to §2 and §3.
3. Add the short name to §1 and to the manifest's allowed platform
   vocabulary.
4. Add a `nvidia_gpu_utils` detector for it and gate pipeline
   construction as needed (see the VP9 parser precedent in
   `savant_deepstream/decoders/src/pipeline.rs`).
