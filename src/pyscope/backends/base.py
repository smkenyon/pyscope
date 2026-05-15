"""Backend protocol shared by all hardware/system readers.

A Backend is constructed once per Monitor lifetime. The sampler calls
`read()` on every tick. Reads must return an iterable of
``(domain, value, kind)`` tuples; the sampler tags each with the current
monotonic timestamp.

Backends are responsible for their own state (counter unwrap, handle
caching). They should not retain references to caller-owned objects.

`kind` strings are open but the canonical set is:
    energy_mj, power_mw, power_mw_estimated, util_pct, bytes, count
"""

from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable

Sample = tuple[str, float, str]  # (domain, value, kind)


@runtime_checkable
class Backend(Protocol):
    name: str

    @classmethod
    def is_available(cls) -> bool: ...

    def read(self) -> Iterable[Sample]: ...

    def close(self) -> None: ...
