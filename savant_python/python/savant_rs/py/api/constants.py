"""Common constants used across the Savant client SDK."""

DEFAULT_NAMESPACE: str = "default"
"""Default attribute namespace."""

DEFAULT_TIME_BASE: tuple[int, int] = (1, 10**9)
"""Default time base (1 nanosecond)."""

DEFAULT_FRAMERATE: str = "30/1"
"""Default framerate as a GStreamer fraction string."""
