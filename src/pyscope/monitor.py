"""Monitor: lifecycle, annotations, and module-level singleton glue.

The Monitor owns a Sampler thread and an EventLog. User code calls
`annotate(label)` for point markers and `with scope(label):` for paired
range markers. Decorator form `@monitor.scoped("foo")` desugars to a scope.

Module-level convenience (`pyscope.start()`, `pyscope.annotate(...)`, etc.)
delegates to a process-wide singleton.

Backend auto-discovery: when `backends=None`, only backends that pass
``is_available()`` are instantiated. Real backends land in stages 4-7;
the `_fake` backend is hidden (underscore-prefixed) and must be passed in
explicitly.
"""

from __future__ import annotations

import functools
import inspect
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, TypeVar

from pyscope.analysis import AnalysisResult
from pyscope.backends.base import Backend
from pyscope.events import EventLog
from pyscope.sampler import Sampler

log = logging.getLogger("pyscope.monitor")

F = TypeVar("F", bound=Callable[..., Any])


def _auto_discover_backends() -> list[Backend]:
    """Try each known backend's `is_available` and instantiate the passes.

    Backends are imported lazily so optional dependencies (`zeus`, `pynvml`,
    `psutil`) don't blow up on environments without them. Failures during
    is_available/init are swallowed with a single warning log so the monitor
    can still run on degraded environments.
    """
    candidates: list[type[Backend]] = []
    try:
        from pyscope.backends.psutil_sys import PsutilSysBackend

        candidates.append(PsutilSysBackend)
    except Exception:
        log.exception("psutil_sys import failed")
    try:
        from pyscope.backends.zeus_cpu import ZeusCpuBackend

        candidates.append(ZeusCpuBackend)
    except Exception:
        log.exception("zeus_cpu import failed")
    try:
        from pyscope.backends.zeus_gpu import ZeusGpuBackend

        candidates.append(ZeusGpuBackend)
    except Exception:
        log.exception("zeus_gpu import failed")
    try:
        from pyscope.backends.nvml_util import NvmlUtilBackend

        candidates.append(NvmlUtilBackend)
    except Exception:
        log.exception("nvml_util import failed")
    try:
        from pyscope.backends.zeus_soc import ZeusSocBackend

        candidates.append(ZeusSocBackend)
    except Exception:
        log.exception("zeus_soc import failed")

    backends: list[Backend] = []
    zeus_cpu_active = False
    for cls in candidates:
        try:
            if not cls.is_available():
                continue
            instance = cls()
            backends.append(instance)
            if getattr(instance, "name", "") == "zeus_cpu":
                zeus_cpu_active = True
        except Exception:
            log.exception("backend %s failed to initialize", cls.__name__)

    # tdp_fallback only activates if no real CPU energy backend is running.
    if not zeus_cpu_active:
        try:
            from pyscope.backends.tdp_fallback import TdpFallbackBackend

            if TdpFallbackBackend.is_available():
                backends.append(TdpFallbackBackend())
        except Exception:
            log.exception("tdp_fallback failed to initialize")

    return backends


def _auto_discover_fanout() -> list[Any]:
    """Try each known fanout emitter; instantiate the ones whose imports succeed."""
    fanouts: list[Any] = []
    for modpath, clsname in [
        ("pyscope.fanout.nvtx_out", "NvtxFanout"),
        ("pyscope.fanout.otel_out", "OtelFanout"),
        # perf_out is opt-in: only useful when the user has created the FIFO,
        # so we don't auto-enable it. Users can pass it explicitly.
    ]:
        try:
            mod = __import__(modpath, fromlist=[clsname])
            cls = getattr(mod, clsname)
            if cls.is_available():
                fanouts.append(cls())
        except Exception:
            log.exception("fanout %s failed to initialize", clsname)
    return fanouts


class Monitor:
    def __init__(
        self,
        interval_ms: int = 50,
        backends: list[Backend] | None = None,
        fanout: list[Any] | None = None,
        output_dir: Path | None = None,
    ) -> None:
        self.interval_ms = int(interval_ms)
        self.output_dir = Path(output_dir) if output_dir else None
        self._explicit_backends = backends
        self._explicit_fanout = fanout
        self.events = EventLog()
        self._sampler: Sampler | None = None
        self._stopped_sampler: Sampler | None = None
        self._wall_clock_anchor_ns: int = 0
        self._monotonic_anchor_ns: int = 0
        self._fanout: list[Any] = []

    @property
    def is_running(self) -> bool:
        return self._sampler is not None

    def start(self) -> "Monitor":
        if self._sampler is not None:
            log.warning("Monitor.start() called while already running; ignoring")
            return self
        backends = (
            list(self._explicit_backends)
            if self._explicit_backends is not None
            else _auto_discover_backends()
        )
        self._fanout = (
            list(self._explicit_fanout)
            if self._explicit_fanout is not None
            else _auto_discover_fanout()
        )
        self._wall_clock_anchor_ns = time.time_ns()
        self._monotonic_anchor_ns = time.monotonic_ns()
        self._sampler = Sampler(backends, self.interval_ms)
        self._sampler.start()
        return self

    def stop(self) -> None:
        if self._sampler is None:
            log.warning("Monitor.stop() called while not running; ignoring")
            return
        self._sampler.stop()
        self._stopped_sampler = self._sampler
        self._sampler = None

    def annotate(self, label: str, **metadata: Any) -> None:
        self.events.point(label, metadata)
        for fo in self._fanout:
            try:
                fo.on_point(label, metadata)
            except Exception:
                log.exception("fanout on_point raised")

    @contextmanager
    def scope(self, label: str, **metadata: Any) -> Iterator[None]:
        self.events.enter(label, metadata)
        for fo in self._fanout:
            try:
                fo.on_enter(label, metadata)
            except Exception:
                log.exception("fanout on_enter raised")
        try:
            yield
        finally:
            self.events.exit(label, metadata)
            for fo in self._fanout:
                try:
                    fo.on_exit(label, metadata)
                except Exception:
                    log.exception("fanout on_exit raised")

    def scoped(self, label: str) -> Callable[[F], F]:
        def decorator(fn: F) -> F:
            if inspect.iscoroutinefunction(fn):
                raise TypeError(
                    "pyscope.scoped does not support coroutine functions in v1"
                )

            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.scope(label):
                    return fn(*args, **kwargs)

            return wrapper  # type: ignore[return-value]

        return decorator

    def analyze(self) -> "AnalysisResult":
        """Build an AnalysisResult from collected samples and events."""
        from pyscope.analysis import build_analysis

        sampler = self._sampler or self._stopped_sampler
        samples = sampler.all_samples() if sampler is not None else []
        return build_analysis(
            samples,
            self.events.snapshot(),
            wall_clock_anchor_ns=self._wall_clock_anchor_ns,
            monotonic_anchor_ns=self._monotonic_anchor_ns,
        )


# --- Module-level singleton ---------------------------------------------

_singleton: Monitor | None = None


def _get_singleton() -> Monitor:
    global _singleton
    if _singleton is None:
        _singleton = Monitor()
    return _singleton


def start(**kwargs: Any) -> Monitor:
    """Start the module-level singleton Monitor. Idempotent."""
    global _singleton
    if _singleton is not None and _singleton.is_running:
        log.warning("pyscope.start() called while already running; ignoring")
        return _singleton
    if _singleton is None or kwargs:
        _singleton = Monitor(**kwargs)
    return _singleton.start()


def stop() -> None:
    if _singleton is None or not _singleton.is_running:
        log.warning("pyscope.stop() called while not running; ignoring")
        return
    _singleton.stop()


def annotate(label: str, **metadata: Any) -> None:
    _get_singleton().annotate(label, **metadata)


@contextmanager
def scope(label: str, **metadata: Any) -> Iterator[None]:
    with _get_singleton().scope(label, **metadata):
        yield


def scoped(label: str) -> Callable[[F], F]:
    return _get_singleton().scoped(label)


def analyze() -> Any:
    return _get_singleton().analyze()


def _reset_singleton_for_tests() -> None:
    """Reset the module-level singleton. Test-only."""
    global _singleton
    if _singleton is not None and _singleton.is_running:
        _singleton.stop()
    _singleton = None
