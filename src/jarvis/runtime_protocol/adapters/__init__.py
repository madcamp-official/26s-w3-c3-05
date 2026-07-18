"""Real Windows and SmartThings adapter implementations."""

from jarvis.runtime_protocol.adapters.base import (
    AdapterResult,
    AdapterStatus,
    DeviceAdapter,
    DispatchCoordinator,
    DispatchReport,
    UnknownAdapterError,
)
from jarvis.runtime_protocol.adapters.http import (
    HttpRequest,
    HttpResponse,
    HttpTransport,
    TransportError,
    TransportNetworkError,
    TransportTimeout,
    UrllibTransport,
)
from jarvis.runtime_protocol.adapters.smartthings import (
    SmartThingsAdapter,
    SmartThingsConfig,
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
    "HttpRequest",
    "HttpResponse",
    "HttpTransport",
    "InputKey",
    "InputSink",
    "SmartThingsAdapter",
    "SmartThingsConfig",
    "TransportError",
    "TransportNetworkError",
    "TransportTimeout",
    "UnknownAdapterError",
    "UrllibTransport",
    "Win32InputSink",
    "WindowsAdapter",
]
