"""Monitor: lifecycle, annotations, and module-level singleton glue.

The Monitor spawns a sampler subprocess (`pyscope.sampler_main`) on
``start()``, hands it an `AF_UNIX SOCK_DGRAM` socket path, and waits for a
``READY\\n`` handshake. Every ``annotate()`` / ``scope`` enter/exit emits one
msgpack datagram so the subprocess can interleave it with timestamped
samples; ``stop()`` sends ``("__stop__",)`` and waits for the subprocess to
flush ``samples.parquet`` + ``events.parquet`` to ``output_dir``.

Annotations are *also* recorded into a local EventLog so user code can
inspect events without round-tripping through parquet (used by tests).
"""

from __future__ import annotations

import atexit
import functools
import inspect
import logging
import os
import select
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, TypeVar

import msgpack

from pyscope.analysis import AnalysisResult
from pyscope.backends import registry
from pyscope.events import EventLog, event_to_tuple

log = logging.getLogger("pyscope.monitor")

F = TypeVar("F", bound=Callable[..., Any])

_READY_TIMEOUT_S = 5.0
_STOP_TIMEOUT_S = 5.0
_MAX_METADATA_BYTES = 16 * 1024


def _auto_discover_backend_names() -> list[str]:
    """Return registered backend names whose `is_available()` is True.

    The fake backend is hidden (underscore-prefixed in spirit) — kept out of
    auto-discovery; pass it in explicitly via ``Monitor(backends=["fake"])``.
    Backends that fail to import or whose `is_available()` raises are
    silently skipped (debug-logged).
    """
    names = [
        "psutil_sys",
        "zeus_cpu",
        "zeus_gpu",
        "nvml_util",
        "zeus_soc",
    ]
    out: list[str] = []
    has_zeus_cpu = False
    for name in names:
        try:
            cls = registry.get(name)
        except KeyError:
            continue
        try:
            if not cls.is_available():
                continue
        except Exception:
            log.debug("is_available raised for %s", name, exc_info=True)
            continue
        out.append(name)
        if name == "zeus_cpu":
            has_zeus_cpu = True
    # tdp_fallback only activates when zeus_cpu isn't available.
    if not has_zeus_cpu:
        try:
            cls = registry.get("tdp_fallback")
            if cls.is_available():
                out.append("tdp_fallback")
        except (KeyError, Exception):
            pass
    return out


def _auto_discover_fanout() -> list[Any]:
    """Try each known fanout emitter; instantiate the ones whose imports succeed."""
    fanouts: list[Any] = []
    for modpath, clsname in [
        ("pyscope.fanout.nvtx_out", "NvtxFanout"),
        ("pyscope.fanout.otel_out", "OtelFanout"),
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
        backends: list[str] | None = None,
        fanout: list[Any] | None = None,
        output_dir: Path | None = None,
    ) -> None:
        self.interval_ms = int(interval_ms)
        self._explicit_output_dir: Path | None = (
            Path(output_dir) if output_dir is not None else None
        )
        self.output_dir: Path | None = None
        self._explicit_backends = backends
        self._explicit_fanout = fanout
        self.events = EventLog()
        self._proc: subprocess.Popen | None = None
        self._sock: socket.socket | None = None
        self._sock_path: str | None = None
        self._wall_clock_anchor_ns: int = 0
        self._monotonic_anchor_ns: int = 0
        self._fanout: list[Any] = []
        self._send_errors_logged: int = 0
        self._owns_tempdir: bool = False
        self._monitor_log_path: Path | None = None

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> "Monitor":
        if self._proc is not None:
            log.warning("Monitor.start() called while already running; ignoring")
            return self
        names = (
            list(self._explicit_backends)
            if self._explicit_backends is not None
            else _auto_discover_backend_names()
        )
        for name in names:
            try:
                registry.get(name)
            except KeyError:
                raise ValueError(
                    f"unknown backend {name!r}; known={registry.available_names()}"
                )
        self._fanout = (
            list(self._explicit_fanout)
            if self._explicit_fanout is not None
            else _auto_discover_fanout()
        )

        if self._explicit_output_dir is not None:
            self.output_dir = self._explicit_output_dir
            self._owns_tempdir = False
        else:
            self.output_dir = Path(tempfile.mkdtemp(prefix="pyscope-"))
            self._owns_tempdir = True
        self.output_dir.mkdir(parents=True, exist_ok=True)

        sock_path = f"/tmp/pyscope-{os.getpid()}-{uuid.uuid4().hex[:8]}.sock"
        self._sock_path = sock_path
        self._monitor_log_path = self.output_dir / "monitor.log"
        log_fh = open(self._monitor_log_path, "w")

        argv = [
            sys.executable,
            "-m",
            "pyscope.sampler_main",
            "--sock", sock_path,
            "--target-pid", str(os.getpid()),
            "--interval-ms", str(self.interval_ms),
            "--output-dir", str(self.output_dir),
            "--backends", ",".join(names) if names else "fake",
        ]
        # NB: stdout=PIPE so we can read the READY line; stderr → monitor.log.
        self._wall_clock_anchor_ns = time.time_ns()
        self._monotonic_anchor_ns = time.monotonic_ns()
        self._proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=log_fh,
        )
        try:
            self._await_ready()
            self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            self._sock.connect(sock_path)
        except Exception:
            self._cleanup_failed_start()
            raise

        atexit.register(self._atexit_stop)
        return self

    def _await_ready(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        end = time.monotonic() + _READY_TIMEOUT_S
        line = b""
        fd = self._proc.stdout.fileno()
        while True:
            remaining = end - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    f"sampler subprocess failed to bind within {_READY_TIMEOUT_S}s; "
                    f"see {self._monitor_log_path}"
                )
            r, _, _ = select.select([fd], [], [], remaining)
            if not r:
                continue
            chunk = os.read(fd, 64)
            if not chunk:
                raise RuntimeError(
                    "sampler subprocess exited before READY; "
                    f"see {self._monitor_log_path}"
                )
            line += chunk
            if b"\n" in line:
                if line.startswith(b"READY"):
                    return
                raise RuntimeError(
                    f"sampler subprocess emitted unexpected handshake: {line!r}"
                )

    def _cleanup_failed_start(self) -> None:
        if self._proc is not None:
            try:
                self._proc.kill()
            except Exception:
                pass
            self._proc = None
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        if self._sock_path and os.path.exists(self._sock_path):
            try:
                os.unlink(self._sock_path)
            except OSError:
                pass

    def stop(self) -> None:
        if self._proc is None:
            log.warning("Monitor.stop() called while not running; ignoring")
            return
        proc = self._proc
        sock = self._sock
        # Send stop, then wait. Best-effort — if send fails the process is
        # already gone and we'll fall through to wait().
        if sock is not None:
            try:
                sock.send(msgpack.packb(("__stop__",), use_bin_type=True))
            except OSError:
                log.debug("send(__stop__) failed; subprocess likely already exiting")
        try:
            proc.wait(timeout=_STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            log.warning("sampler subprocess did not exit within %ss; killing", _STOP_TIMEOUT_S)
            proc.kill()
            proc.wait(timeout=2.0)
        finally:
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass
            self._sock = None
            self._proc = None
            if self._sock_path and os.path.exists(self._sock_path):
                try:
                    os.unlink(self._sock_path)
                except OSError:
                    pass
            if proc.returncode not in (0, None):
                if self._monitor_log_path and self._monitor_log_path.exists():
                    sys.stderr.write(
                        f"pyscope: sampler subprocess exited with code "
                        f"{proc.returncode}; see {self._monitor_log_path}\n"
                    )

    def _atexit_stop(self) -> None:
        if self._proc is not None:
            try:
                self.stop()
            except Exception:
                pass
        if self._owns_tempdir and self.output_dir is not None:
            # Tempdir cleanup only on clean exit (no Python exception in flight).
            if sys.exc_info() == (None, None, None):
                try:
                    shutil.rmtree(self.output_dir, ignore_errors=True)
                except Exception:
                    pass

    # --- Event API ----------------------------------------------------

    def _send_event(self, ts_ns: int, label: str, role: str, metadata: dict) -> None:
        if self._sock is None:
            return
        # Cap metadata serialized size.
        meta = metadata
        try:
            packed = msgpack.packb(
                (ts_ns, label, role, meta, 0),  # thread_id filled below
                use_bin_type=True,
                default=str,
            )
            if len(packed) > _MAX_METADATA_BYTES:
                log.warning(
                    "annotation %r metadata too large (%d bytes); dropping metadata",
                    label, len(packed),
                )
                meta = {}
        except Exception:
            meta = {}
        import threading
        tid = threading.get_ident()
        try:
            self._sock.send(
                msgpack.packb(
                    (ts_ns, label, role, meta, tid), use_bin_type=True, default=str
                )
            )
        except OSError as e:
            if self._send_errors_logged < 1:
                log.warning("annotation send failed (%s); subprocess may have died", e)
                self._send_errors_logged += 1

    def annotate(self, label: str, **metadata: Any) -> None:
        evt = self.events.point(label, metadata)
        self._send_event(evt.ts_ns, evt.label, evt.role, evt.metadata)
        for fo in self._fanout:
            try:
                fo.on_point(label, metadata)
            except Exception:
                log.exception("fanout on_point raised")

    @contextmanager
    def scope(self, label: str, **metadata: Any) -> Iterator[None]:
        evt_enter = self.events.enter(label, metadata)
        self._send_event(evt_enter.ts_ns, evt_enter.label, evt_enter.role, evt_enter.metadata)
        for fo in self._fanout:
            try:
                fo.on_enter(label, metadata)
            except Exception:
                log.exception("fanout on_enter raised")
        try:
            yield
        finally:
            evt_exit = self.events.exit(label, metadata)
            self._send_event(evt_exit.ts_ns, evt_exit.label, evt_exit.role, evt_exit.metadata)
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

    # --- Analysis -----------------------------------------------------

    def analyze(self) -> "AnalysisResult":
        """Build an AnalysisResult from collected samples + local events."""
        from pyscope.analysis import build_analysis

        raw_samples: list[tuple[int, str, str, float, str]] = []
        if self.output_dir is not None:
            samples_path = self.output_dir / "samples.parquet"
            if samples_path.exists():
                try:
                    import polars as pl

                    df = pl.read_parquet(samples_path)
                    raw_samples = [
                        (int(r[0]), str(r[1]), str(r[2]), float(r[3]), str(r[4]))
                        for r in df.iter_rows()
                    ]
                except Exception:
                    log.exception("failed to read samples.parquet from %s", samples_path)
        return build_analysis(
            raw_samples,
            self.events.snapshot(),
            wall_clock_anchor_ns=self._wall_clock_anchor_ns,
            monotonic_anchor_ns=self._monotonic_anchor_ns,
        )

    # Unused references — kept for compatibility/refs in tests.
    _ = event_to_tuple  # silence unused-import on mypy


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
