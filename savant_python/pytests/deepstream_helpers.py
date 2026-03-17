"""Shared helpers for DeepStream GPU tests.

Consolidates generator factories and memory-leak test utilities used
across ``test_deepstream_batched.py``, ``test_deepstream_heterogeneous.py``,
``test_deepstream_transform.py``, and others.
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import savant_rs.deepstream


# ─── Generator factory ─────────────────────────────────────────────────


def make_gen(
    fmt: str,
    w: int,
    h: int,
    pool_size: int = 4,
) -> "savant_rs.deepstream.DsNvSurfaceBufferGenerator":
    """Create a ``DsNvSurfaceBufferGenerator`` with given parameters."""
    import savant_rs.deepstream as ds

    return ds.DsNvSurfaceBufferGenerator(fmt, w, h, pool_size=pool_size)


# ─── Memory-leak test helpers ──────────────────────────────────────────

LEAK_ITERATIONS = 200
WARMUP_ITERATIONS = 10
CPU_GROWTH_LIMIT_KB = 10_000  # 10 MB
GPU_GROWTH_LIMIT_MB = 20


def gpu_mem_used_mb() -> int:
    """Query current GPU memory usage (MiB)."""
    import savant_rs.deepstream as ds

    return ds.gpu_mem_used_mib()


def cpu_rss_kb() -> int:
    """Current process RSS in KB (Linux)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except Exception:
        pass
    return 0


def assert_no_leak(
    label: str,
    cpu_before: int,
    cpu_after: int,
    gpu_before: int,
    gpu_after: int,
) -> None:
    """Assert that CPU and GPU memory growth stayed within limits."""
    cpu_growth = cpu_after - cpu_before
    gpu_growth = gpu_after - gpu_before
    assert cpu_growth < CPU_GROWTH_LIMIT_KB, (
        f"{label}: CPU RSS grew by {cpu_growth} KB over {LEAK_ITERATIONS} iterations"
    )
    assert gpu_growth < GPU_GROWTH_LIMIT_MB, (
        f"{label}: GPU mem grew by {gpu_growth} MB over {LEAK_ITERATIONS} iterations"
    )


def run_leak_test(warmup_fn, body_fn, label: str = "leak test") -> None:
    """Run a warmup + measurement cycle and assert no memory leak.

    Args:
        warmup_fn: Called ``WARMUP_ITERATIONS`` times (no measurement).
        body_fn:   Called ``LEAK_ITERATIONS`` times (measured).
        label:     Description for assertion messages.
    """
    for _ in range(WARMUP_ITERATIONS):
        warmup_fn()

    gc.collect()
    cpu_before = cpu_rss_kb()
    gpu_before = gpu_mem_used_mb()

    for i in range(LEAK_ITERATIONS):
        body_fn()

    gc.collect()
    assert_no_leak(
        label,
        cpu_before, cpu_rss_kb(),
        gpu_before, gpu_mem_used_mb(),
    )
