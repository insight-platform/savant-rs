from enum import Enum
from typing import Dict, Optional

__all__ = [
    "LogLevel",
    "set_log_level",
    "get_log_level",
    "log_level_enabled",
    "log",
]

class LogLevel(Enum):
    Trace: int
    Debug: int
    Info: int
    Warning: int
    Error: int
    Off: int

def set_log_level(level: LogLevel) -> LogLevel: ...
def get_log_level() -> LogLevel: ...
def log_level_enabled(level: LogLevel) -> bool: ...
def log(
    level: LogLevel,
    target: str,
    message: str,
    params: Optional[Dict[str, str]] = None,
    no_gil: bool = True,
) -> None: ...
