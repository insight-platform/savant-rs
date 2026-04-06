"""Type stubs for ``savant_rs.retina_rtsp``.

Available only when ``savant_rs`` is built with the ``gst`` Cargo feature.
"""

from typing import List, Optional

__all__ = [
    "RtspBackend",
    "RtspSourceOptions",
    "RtspSource",
    "SyncConfiguration",
    "RtspSourceGroup",
    "RetinaRtspService",
]

class RtspBackend:
    Retina: RtspBackend
    Gstreamer: RtspBackend

class RtspSourceOptions:
    """RTSP authentication credentials."""

    def __init__(self, username: str, password: str) -> None: ...

    @property
    def username(self) -> str: ...
    @property
    def password(self) -> str: ...

class RtspSource:
    """A single RTSP source within a group."""

    def __init__(
        self,
        source_id: str,
        url: str,
        stream_position: Optional[int] = None,
        options: Optional[RtspSourceOptions] = None,
    ) -> None: ...

    @property
    def source_id(self) -> str: ...
    @property
    def url(self) -> str: ...
    @property
    def stream_position(self) -> Optional[int]: ...
    @property
    def options(self) -> Optional[RtspSourceOptions]: ...

class SyncConfiguration:
    """RTCP SR synchronisation parameters."""

    def __init__(
        self,
        group_window_duration_ms: int,
        batch_duration_ms: int,
        network_skew_correction: bool = False,
        rtcp_once: bool = False,
    ) -> None: ...

    @property
    def group_window_duration_ms(self) -> int: ...
    @property
    def batch_duration_ms(self) -> int: ...
    @property
    def network_skew_correction(self) -> bool: ...
    @property
    def rtcp_once(self) -> bool: ...

class RtspSourceGroup:
    """A group of RTSP sources sharing a backend and optional sync config."""

    def __init__(
        self,
        sources: List[RtspSource],
        backend: RtspBackend = ...,
        rtcp_sr_sync: Optional[SyncConfiguration] = None,
    ) -> None: ...

    @property
    def sources(self) -> List[RtspSource]: ...
    @property
    def backend(self) -> RtspBackend: ...
    @property
    def rtcp_sr_sync(self) -> Optional[SyncConfiguration]: ...

class RetinaRtspService:
    """Stateful RTSP ingestion service with a shared sink socket.

    Create from a JSON configuration file.  Use :meth:`run_group` (from a
    Python thread) to start a group and :meth:`stop_group` to stop it.
    """

    def __init__(self, config_path: str) -> None:
        """Load configuration and open the ZeroMQ sink socket.

        Args:
            config_path: Path to the JSON configuration file.

        Raises:
            RuntimeError: On invalid config or socket error.
        """
        ...

    def run_group(self, group: RtspSourceGroup, name: str) -> None:
        """Run a source group.  Blocks (GIL released) until stopped.

        Intended to be called from a dedicated ``threading.Thread``.
        Automatically reconnects on transient failures.

        Args:
            group: The group configuration.
            name: Unique name for this group.

        Raises:
            RuntimeError: On unrecoverable errors.
        """
        ...

    def stop_group(self, name: str) -> None:
        """Stop a running group.  Blocks (GIL released) until finished.

        Args:
            name: Name passed to :meth:`run_group`.
        """
        ...

    def shutdown(self) -> None:
        """Stop all running groups.  Blocks (GIL released)."""
        ...

    @property
    def running_groups(self) -> List[str]:
        """Names of currently running groups."""
        ...
