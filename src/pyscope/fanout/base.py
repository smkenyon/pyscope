"""Fanout protocol.

Fanouts mirror pyscope's annotations into other observability systems
(NVTX, OpenTelemetry, perf, etc.) so a single set of cursor markers shows
up in every tool a user already uses.

All fanout calls are best-effort: the Monitor wraps each callback in
try/except so a broken fanout cannot break user code.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Fanout(Protocol):
    name: str

    @classmethod
    def is_available(cls) -> bool: ...

    def on_point(self, label: str, metadata: dict) -> None: ...

    def on_enter(self, label: str, metadata: dict) -> None: ...

    def on_exit(self, label: str, metadata: dict) -> None: ...
