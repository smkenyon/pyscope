"""Thread-less sample loop used by the subprocess sampler.

`SampleLoop` owns one `BackendBuffer` per backend, ticks at a fixed interval,
and tolerates per-backend failure (drops a backend after MAX_BACKEND_FAILS
consecutive errors). Drift correction: tick N targets
``start_monotonic_ns + N * interval_ns``; on overrun we skip ticks rather
than play catch-up.

This module no longer spawns a thread — the sampler subprocess
(`pyscope.sampler_main`) drives it on its own process's main thread, which
is why the GIL no longer matters.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from pyscope.backends.base import Backend

log = logging.getLogger("pyscope.sampler")

MAX_BACKEND_FAILS = 5


@dataclass
class BackendBuffer:
    backend: Backend
    samples: list[tuple[int, str, str, float, str]] = field(default_factory=list)
    consecutive_fails: int = 0
    dropped: bool = False


class SampleLoop:
    """Drift-corrected polling loop. Caller drives it via ``tick_if_due()``."""

    def __init__(self, backends: list[Backend], interval_ms: int) -> None:
        if interval_ms <= 0:
            raise ValueError("interval_ms must be positive")
        self.interval_ns: int = int(interval_ms) * 1_000_000
        self.buffers: list[BackendBuffer] = [BackendBuffer(b) for b in backends]
        self._start_monotonic_ns: int = 0
        self._tick_n: int = 0
        self._skipped_ticks_logged: bool = False
        self._started: bool = False

    def start(self) -> None:
        self._start_monotonic_ns = time.monotonic_ns()
        self._started = True

    @property
    def start_monotonic_ns(self) -> int:
        return self._start_monotonic_ns

    def next_tick_ns(self) -> int:
        """Absolute monotonic_ns at which the next tick should fire."""
        return self._start_monotonic_ns + self._tick_n * self.interval_ns

    def tick_if_due(self, now_ns: int | None = None) -> bool:
        """Sample once if the next scheduled tick is due. Returns True if it fired."""
        if not self._started:
            raise RuntimeError("SampleLoop.start() must be called first")
        if now_ns is None:
            now_ns = time.monotonic_ns()
        target = self.next_tick_ns()
        if now_ns < target:
            return False
        # If we're more than one interval behind, skip ahead instead of
        # playing catch-up (logged once).
        if now_ns - target >= self.interval_ns:
            missed = (now_ns - target) // self.interval_ns
            if missed > 0 and not self._skipped_ticks_logged:
                log.warning(
                    "sampler running behind: skipping %d ticks at interval=%d ns",
                    missed,
                    self.interval_ns,
                )
                self._skipped_ticks_logged = True
            self._tick_n += int(missed)
        self._sample_once(now_ns)
        self._tick_n += 1
        return True

    def close(self) -> None:
        for buf in self.buffers:
            try:
                buf.backend.close()
            except Exception:
                log.exception("backend %s.close() raised", buf.backend.name)

    def all_samples(self) -> list[tuple[int, str, str, float, str]]:
        out: list[tuple[int, str, str, float, str]] = []
        for buf in self.buffers:
            out.extend(buf.samples)
        out.sort(key=lambda row: row[0])
        return out

    def _sample_once(self, ts_ns: int) -> None:
        for buf in self.buffers:
            if buf.dropped:
                continue
            try:
                samples = list(buf.backend.read())
            except Exception:
                buf.consecutive_fails += 1
                log.exception(
                    "backend %s.read() raised (%d/%d)",
                    buf.backend.name,
                    buf.consecutive_fails,
                    MAX_BACKEND_FAILS,
                )
                if buf.consecutive_fails >= MAX_BACKEND_FAILS:
                    buf.dropped = True
                    log.error(
                        "backend %s disabled after %d consecutive failures",
                        buf.backend.name,
                        MAX_BACKEND_FAILS,
                    )
                continue
            buf.consecutive_fails = 0
            for domain, value, kind in samples:
                buf.samples.append((ts_ns, buf.backend.name, domain, float(value), kind))
