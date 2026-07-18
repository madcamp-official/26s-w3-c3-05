"""Real Windows and SmartThings adapter implementations."""

from jarvis.runtime_protocol.adapters.base import (
    AdapterResult,
    AdapterStatus,
    DeviceAdapter,
    DispatchCoordinator,
    DispatchReport,
    UnknownAdapterError,
)
from jarvis.runtime_protocol.adapters.windows import (
    InputKey,
    InputSink,
    Win32InputSink,
    WindowsAdapter,
)

__all__ = [
    "AdapterResult",
    "AdapterStatus",
    "DeviceAdapter",
    "DispatchCoordinator",
    "DispatchReport",
    "InputKey",
    "InputSink",
    "UnknownAdapterError",
    "Win32InputSink",
    "WindowsAdapter",
]
