__all__ = [
    "StreamProperties",
    "StreamInfoVideoFile",
    "StreamInfoRTSP",
]

class StreamProperties:
    def __init__(self): ...

class StreamInfoVideoFile:
    path: str
    looped: bool
    sync: bool

    def __init__(self, path: str, looped: bool, sync: bool): ...

class StreamInfoRTSP:
    url: str
    tcp: bool

    def __init__(self, url: str, tcp: bool): ...