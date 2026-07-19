"""Device capability model and value validation (README 10장).

Devices are described by what they can do, not by brand. A device profile lists
named capabilities; each capability declares its type, the operations it accepts,
and (for numbers) its range and step. The protocol validates an incoming Intent's
``capability``/``operation``/``value`` against this model *before* a command is
ever dispatched (development-principles 2.5) so out-of-range or unsupported
requests never reach an adapter.

MVP scope covers boolean and number capabilities (power, brightness,
color_temperature). Enum capabilities (e.g. an aircon ``mode``) are deferred:
the ``Intent.value`` contract is ``int | float | bool`` and cannot yet carry an
enum member string. See documents/runtime-protocol.md 이슈.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

# Values on the module boundary (Intent/Command) are these primitives only.
CommandValue = int | float | bool

_STEP_TOLERANCE = 1e-9


class Operation(StrEnum):
    """Operations a capability may accept. Open by config, named here for reuse."""

    SET = "set"
    INCREMENT = "increment"
    DECREMENT = "decrement"
    TOGGLE = "toggle"


@dataclass(frozen=True, slots=True)
class BooleanCapability:
    """An on/off capability such as ``power``."""

    operations: frozenset[str] = frozenset({Operation.SET, Operation.TOGGLE})


@dataclass(frozen=True, slots=True)
class NumberCapability:
    """A bounded numeric capability such as ``brightness`` on a ``[min,max]`` grid."""

    minimum: float
    maximum: float
    step: float
    operations: frozenset[str] = frozenset(
        {Operation.SET, Operation.INCREMENT, Operation.DECREMENT}
    )

    def __post_init__(self) -> None:
        if self.step <= 0:
            raise ValueError(f"step must be > 0, got {self.step}")
        if self.minimum > self.maximum:
            raise ValueError(
                f"minimum {self.minimum} exceeds maximum {self.maximum}"
            )


Capability = BooleanCapability | NumberCapability


@dataclass(frozen=True, slots=True)
class DeviceProfile:
    """A registered device: its adapter and named capabilities."""

    device_id: str
    adapter: str
    capabilities: Mapping[str, Capability]


class DeviceRegistry:
    """Lookup of registered devices by id.

    Built from config-shaped data by Runtime; consumers never mutate it. An
    unregistered device id yields ``None`` so callers reject rather than guess.
    """

    def __init__(self, profiles: Mapping[str, DeviceProfile]) -> None:
        self._profiles = dict(profiles)

    def get(self, device_id: str) -> DeviceProfile | None:
        return self._profiles.get(device_id)

    def __contains__(self, device_id: object) -> bool:
        return device_id in self._profiles


@dataclass(frozen=True, slots=True)
class ValidationFailure:
    """Why a value/operation is invalid, for trace and rejection detail."""

    detail: str


def validate_request(
    capability: Capability, operation: str, value: CommandValue
) -> ValidationFailure | None:
    """Validate an operation+value against a capability spec.

    Returns ``None`` when acceptable, otherwise a :class:`ValidationFailure` describing
    the problem. For relative number operations (increment/decrement) only the
    delta is validated here; clamping the resulting absolute value to ``[min,max]``
    needs live device state and is the adapter's responsibility at apply time.
    """
    if operation not in capability.operations:
        return ValidationFailure(f"operation {operation!r} not supported by capability")

    if isinstance(capability, BooleanCapability):
        return _validate_boolean(operation, value)
    return _validate_number(capability, operation, value)


def _validate_boolean(operation: str, value: CommandValue) -> ValidationFailure | None:
    if operation == Operation.TOGGLE:
        return None  # value is ignored for toggle
    if not isinstance(value, bool):
        return ValidationFailure(f"boolean 'set' requires a bool value, got {type(value).__name__}")
    return None


def _validate_number(
    capability: NumberCapability, operation: str, value: CommandValue
) -> ValidationFailure | None:
    # bool is a subclass of int; a numeric capability must not accept True/False.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return ValidationFailure(f"number capability requires a numeric value, got {type(value).__name__}")

    numeric = float(value)
    if operation == Operation.SET:
        if not capability.minimum <= numeric <= capability.maximum:
            return ValidationFailure(
                f"value {numeric} outside [{capability.minimum}, {capability.maximum}]"
            )
        if not _on_step_grid(numeric - capability.minimum, capability.step):
            return ValidationFailure(f"value {numeric} not on step grid of {capability.step}")
        return None

    # increment / decrement: value is a positive delta that is a multiple of step.
    if numeric <= 0:
        return ValidationFailure(f"{operation} delta must be positive, got {numeric}")
    if not _on_step_grid(numeric, capability.step):
        return ValidationFailure(f"{operation} delta {numeric} not a multiple of step {capability.step}")
    return None


def _on_step_grid(offset: float, step: float) -> bool:
    steps = round(offset / step)
    return math.isclose(steps * step, offset, abs_tol=_STEP_TOLERANCE)
