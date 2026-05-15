"""The sampler thread.

One thread per Monitor. On every tick it calls `read()` on each backend and
appends ``(ts_ns, source, domain, value, kind)`` tuples to per-backend lists.
Lists are Python lists; `list.append` is atomic under the GIL, and analysis
runs only after `stop()` joins the thread.

Drift correction: tick N fires at ``start_monotonic_ns + N * interval_ns``.
On overrun, we skip ticks rather than play catch-up (logged once).

Per-backend failure isolation: if a backend raises in `read()`, we log once
and increment a fail counter. After ``MAX_BACKEND_FAILS`` consecutive
failures the backend is dropped from the rotation for the rest of the run.
"""

from __future__ import annotations

import logging
import threading
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


class Sampler:
    def __init__(self, backends: list[Backend], interval_ms: int) -> None:
        if interval_ms <= 0:
            raise ValueError("interval_ms must be positive")
        self.interval_ns: int = int(interval_ms) * 1_000_000
        self.buffers: list[BackendBuffer] = [BackendBuffer(b) for b in backends]
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_monotonic_ns: int = 0
        self._skipped_ticks_logged: bool = False

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Sampler already started")
        self._start_monotonic_ns = time.monotonic_ns()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="pyscope-sampler", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=timeout)
        self._thread = None
        for buf in self.buffers:
            try:
                buf.backend.close()
            except Exception:
                log.exception("backend %s.close() raised", buf.backend.name)

    @property
    def start_monotonic_ns(self) -> int:
        return self._start_monotonic_ns

    def all_samples(self) -> list[tuple[int, str, str, float, str]]:
        """Flatten per-backend sample lists. Call after stop()."""
        out: list[tuple[int, str, str, float, str]] = []
        for buf in self.buffers:
            out.extend(buf.samples)
        out.sort(key=lambda row: row[0])
        return out

    def _run(self) -> None:
        n = 0
        while not self._stop.is_set():
            target_ns = self._start_monotonic_ns + n * self.interval_ns
            now = time.monotonic_ns()
            wait_s = (target_ns - now) / 1e9
            if wait_s > 0:
                # threading.Event.wait returns True on set; we want to bail out fast.
                if self._stop.wait(wait_s):
                    break
            else:
                # We're already past the target. Count missed ticks once.
                missed = max(0, (now - target_ns) // self.interval_ns)
                if missed > 0 and not self._skipped_ticks_logged:
                    log.warning(
                        "sampler running behind: skipping %d ticks at interval=%d ns",
                        missed,
                        self.interval_ns,
                    )
                    self._skipped_ticks_logged = True
                n += int(missed)

            ts_ns = time.monotonic_ns()
            self._sample_once(ts_ns)
            n += 1

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
