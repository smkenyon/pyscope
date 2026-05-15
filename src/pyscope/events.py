"""Event records and a thread-safe event log.

Events come from user threads (`annotate`, `scope` enter/exit) and are read
back during analysis after the monitor has stopped. Appends are guarded by a
single lock; reads happen after `stop()` so no lock contention there.

Wall-clock anchoring: timestamps are `time.monotonic_ns()` everywhere. A
single `time.time_ns()` anchor is captured by the Monitor at start for
display purposes only.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["point", "enter", "exit"]


@dataclass(frozen=True)
class Event:
    ts_ns: int
    label: str
    role: Role
    metadata: dict[str, Any] = field(default_factory=dict)
    thread_id: int = 0


class EventLog:
    """Append-only log of events. Thread-safe append; unsynchronized snapshot.

    `snapshot()` returns a shallow copy of the underlying list and is intended
    to be called after the sampler has stopped (i.e. no concurrent appends).
    """

    def __init__(self) -> None:
        self._events: list[Event] = []
        self._lock = threading.Lock()

    def append(self, event: Event) -> None:
        with self._lock:
            self._events.append(event)

    def point(self, label: str, metadata: dict[str, Any] | None = None) -> Event:
        evt = Event(
            ts_ns=time.monotonic_ns(),
            label=label,
            role="point",
            metadata=dict(metadata) if metadata else {},
            thread_id=threading.get_ident(),
        )
        self.append(evt)
        return evt

    def enter(self, label: str, metadata: dict[str, Any] | None = None) -> Event:
        evt = Event(
            ts_ns=time.monotonic_ns(),
            label=label,
            role="enter",
            metadata=dict(metadata) if metadata else {},
            thread_id=threading.get_ident(),
        )
        self.append(evt)
        return evt

    def exit(self, label: str, metadata: dict[str, Any] | None = None) -> Event:
        evt = Event(
            ts_ns=time.monotonic_ns(),
            label=label,
            role="exit",
            metadata=dict(metadata) if metadata else {},
            thread_id=threading.get_ident(),
        )
        self.append(evt)
        return evt

    def snapshot(self) -> list[Event]:
        return list(self._events)

    def __len__(self) -> int:
        return len(self._events)


def event_to_tuple(e: Event) -> tuple[int, str, str, dict[str, Any], int]:
    """Wire format for msgpack transport between Monitor and sampler subprocess."""
    return (e.ts_ns, e.label, e.role, dict(e.metadata), e.thread_id)


def event_from_tuple(t: tuple) -> Event:
    ts_ns, label, role, metadata, thread_id = t
    return Event(
        ts_ns=int(ts_ns),
        label=str(label),
        role=role,  # type: ignore[arg-type]
        metadata=dict(metadata) if metadata else {},
        thread_id=int(thread_id),
    )
