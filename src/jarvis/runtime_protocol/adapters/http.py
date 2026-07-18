"""Minimal HTTP transport boundary for cloud adapters.

Adapters depend on the :class:`HttpTransport` Protocol, never on a concrete HTTP
library, so their request-building and response-handling logic is tested with a
fake transport and no network. :class:`UrllibTransport` is the real boundary
(stdlib ``urllib``); it classifies failures into typed errors so callers can tell
a timeout from a network error (development-principles 6.4).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True, slots=True)
class HttpRequest:
    method: str
    url: str
    headers: Mapping[str, str] = field(default_factory=dict)
    body: bytes | None = None


@dataclass(frozen=True, slots=True)
class HttpResponse:
    status: int
    body: bytes


class TransportError(Exception):
    """Base class for transport-level failures (no HTTP response received)."""


class TransportTimeout(TransportError):
    """The request did not complete within the timeout."""


class TransportNetworkError(TransportError):
    """The request failed to reach the host (DNS, connection refused, ...)."""


class HttpTransport(Protocol):
    def send(self, request: HttpRequest, timeout_s: float) -> HttpResponse:
        """Send a request and return the response.

        Raises :class:`TransportTimeout` or :class:`TransportNetworkError` when no
        response is received. An HTTP error status (4xx/5xx) is a normal return —
        the status is on the response for the caller to classify.
        """
        ...


class UrllibTransport:
    """Real HTTP transport over the standard library (Windows/any host)."""

    def send(self, request: HttpRequest, timeout_s: float) -> HttpResponse:
        import urllib.error
        import urllib.request

        req = urllib.request.Request(
            url=request.url,
            method=request.method,
            headers=dict(request.headers),
            data=request.body,
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as response:
                return HttpResponse(status=response.status, body=response.read())
        except urllib.error.HTTPError as exc:
            # An HTTP error status is a valid response, not a transport failure.
            return HttpResponse(status=exc.code, body=exc.read())
        except TimeoutError as exc:
            raise TransportTimeout(str(exc)) from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise TransportTimeout(str(exc.reason)) from exc
            raise TransportNetworkError(str(exc.reason)) from exc
