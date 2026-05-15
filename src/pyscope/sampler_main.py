"""Subprocess sampler entry. Spawned by ``Monitor.start()``.

Usage:
    python -m pyscope.sampler_main \
        --sock /tmp/pyscope-<pid>.sock \
        --target-pid <pid> \
        --interval-ms 50 \
        --output-dir DIR \
        --backends fake,psutil_sys

The subprocess binds the datagram socket, prints ``READY\\n`` to stdout, and
then loops between draining incoming msgpack events (one datagram per event)
and ticking the SampleLoop. On ``("__stop__",)`` it flushes samples.parquet
and events.parquet into ``--output-dir`` and exits 0.

stderr (including all logging.* output) is captured by the parent into
``<output_dir>/monitor.log``.
"""

from __future__ import annotations

import argparse
import logging
import os
import selectors
import socket
import sys
import time
from pathlib import Path

import msgpack

from pyscope.analysis import events_to_df, samples_to_df
from pyscope.backends import registry
from pyscope.events import Event, event_from_tuple
from pyscope.sampler import SampleLoop

log = logging.getLogger("pyscope.sampler_main")

# Conservatively large datagram buffer. Annotations are usually < 1 KB; the
# README documents a 16 KB metadata cap on the parent side.
RECV_BUFSIZE = 65536


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="pyscope.sampler_main")
    p.add_argument("--sock", required=True)
    p.add_argument("--target-pid", type=int, required=True)
    p.add_argument("--interval-ms", type=int, required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--backends", required=True, help="comma-separated backend names")
    p.add_argument("--log-level", default="WARNING")
    return p.parse_args(argv)


def _build_backends(names: list[str], target_pid: int) -> list:
    out = []
    for name in names:
        try:
            out.append(registry.construct(name, target_pid=target_pid))
        except Exception:
            log.exception("backend %s failed to initialize in subprocess", name)
    return out


def _bind_socket(sock_path: str) -> socket.socket:
    if os.path.exists(sock_path):
        os.unlink(sock_path)
    s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    s.bind(sock_path)
    s.setblocking(False)
    return s


def _drain_events(sock: socket.socket, events: list[Event]) -> bool:
    """Drain all pending datagrams. Returns True iff a __stop__ was seen."""
    stop = False
    while True:
        try:
            data, _ = sock.recvfrom(RECV_BUFSIZE)
        except BlockingIOError:
            return stop
        except OSError:
            return stop
        try:
            parsed = msgpack.unpackb(data, raw=False)
        except Exception:
            log.warning("dropping malformed datagram (%d bytes)", len(data))
            continue
        if isinstance(parsed, (list, tuple)) and len(parsed) >= 1 and parsed[0] == "__stop__":
            stop = True
            continue
        try:
            events.append(event_from_tuple(parsed))
        except Exception:
            log.warning("dropping malformed event: %r", parsed, exc_info=True)


def _flush_parquet(output_dir: Path, samples: list, events: list[Event]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_df = samples_to_df(samples).sort("ts_ns") if samples else samples_to_df([])
    samples_df.write_parquet(output_dir / "samples.parquet")
    events_df = events_to_df(events).sort("ts_ns") if events else events_to_df([])
    events_df.write_parquet(output_dir / "events.parquet")
    log.info(
        "flushed %d samples / %d events to %s",
        samples_df.height, events_df.height, output_dir,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = [n.strip() for n in args.backends.split(",") if n.strip()]
    backends = _build_backends(names, args.target_pid)
    log.info("backends active: %s", [b.name for b in backends])

    sock = _bind_socket(args.sock)
    loop = SampleLoop(backends, args.interval_ms)
    events: list[Event] = []

    # Signal readiness AFTER bind so the parent can connect without racing.
    sys.stdout.write("READY\n")
    sys.stdout.flush()

    sel = selectors.DefaultSelector()
    sel.register(sock, selectors.EVENT_READ)

    loop.start()
    stop_requested = False
    parent_dead_since: float | None = None
    try:
        while not stop_requested:
            # Wait until either a datagram arrives or the next tick is due.
            now_ns = time.monotonic_ns()
            wait_s = max(0.0, (loop.next_tick_ns() - now_ns) / 1e9)
            sel.select(timeout=wait_s)
            if _drain_events(sock, events):
                stop_requested = True
            loop.tick_if_due()

            # Watchdog: if the parent has been dead for >2 intervals, bail
            # so we don't leave the subprocess wandering forever.
            if not _parent_alive(args.target_pid):
                if parent_dead_since is None:
                    parent_dead_since = time.monotonic()
                elif time.monotonic() - parent_dead_since > max(2.0, 5 * args.interval_ms / 1000):
                    log.warning("parent pid %d gone; exiting", args.target_pid)
                    stop_requested = True
            else:
                parent_dead_since = None
    finally:
        loop.close()
        try:
            sock.close()
        except Exception:
            pass
        try:
            os.unlink(args.sock)
        except OSError:
            pass
        _flush_parquet(output_dir, loop.all_samples(), events)
    return 0


def _parent_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


if __name__ == "__main__":
    sys.exit(main())
