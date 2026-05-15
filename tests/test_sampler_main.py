"""End-to-end test for the subprocess sampler entry point.

Spawns `python -m pyscope.sampler_main` with the fake backend, sends a couple
of msgpack events, then __stop__, and asserts the resulting parquet files.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import uuid

import msgpack
import polars as pl


def _await_ready(proc: subprocess.Popen, timeout: float = 5.0) -> None:
    assert proc.stdout is not None
    end = time.monotonic() + timeout
    line = b""
    fd = proc.stdout.fileno()
    while time.monotonic() < end:
        import select as sysselect
        r, _, _ = sysselect.select([fd], [], [], end - time.monotonic())
        if not r:
            continue
        chunk = os.read(fd, 64)
        if not chunk:
            raise RuntimeError("subprocess exited before READY")
        line += chunk
        if b"\n" in line:
            if line.startswith(b"READY"):
                return
            raise RuntimeError(f"unexpected handshake: {line!r}")
    raise TimeoutError("sampler subprocess did not signal READY in time")


def test_subprocess_sampler_roundtrip(tmp_path):
    sock_path = f"/tmp/pyscope-test-{uuid.uuid4().hex[:8]}.sock"
    log_path = tmp_path / "monitor.log"
    argv = [
        sys.executable,
        "-m", "pyscope.sampler_main",
        "--sock", sock_path,
        "--target-pid", str(os.getpid()),
        "--interval-ms", "20",
        "--output-dir", str(tmp_path),
        "--backends", "fake",
        "--log-level", "INFO",
    ]
    with open(log_path, "w") as log_fh:
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=log_fh)
    try:
        _await_ready(proc, timeout=5.0)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        sock.connect(sock_path)

        # Send a couple of events.
        t0 = time.monotonic_ns()
        sock.send(msgpack.packb((t0, "first", "point", {"k": 1}, 0), use_bin_type=True))
        sock.send(msgpack.packb((t0 + 1, "blk", "enter", {}, 0), use_bin_type=True))
        time.sleep(0.15)  # let several ticks accumulate
        sock.send(msgpack.packb((time.monotonic_ns(), "blk", "exit", {}, 0), use_bin_type=True))
        sock.send(msgpack.packb(("__stop__",), use_bin_type=True))

        rc = proc.wait(timeout=5.0)
        assert rc == 0, log_path.read_text()
        sock.close()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=2.0)
        if os.path.exists(sock_path):
            try:
                os.unlink(sock_path)
            except OSError:
                pass

    samples = pl.read_parquet(tmp_path / "samples.parquet")
    events = pl.read_parquet(tmp_path / "events.parquet")
    assert samples.height > 0
    assert "fake_energy_mj" in samples["domain"].unique().to_list()
    labels = events["label"].to_list()
    assert "first" in labels
    assert "blk" in labels


def test_subprocess_unknown_backend_logs_and_continues(tmp_path):
    """Unknown backend names log + skip; subprocess still produces a parquet."""
    sock_path = f"/tmp/pyscope-test-{uuid.uuid4().hex[:8]}.sock"
    log_path = tmp_path / "monitor.log"
    argv = [
        sys.executable,
        "-m", "pyscope.sampler_main",
        "--sock", sock_path,
        "--target-pid", str(os.getpid()),
        "--interval-ms", "20",
        "--output-dir", str(tmp_path),
        "--backends", "fake,does_not_exist",
    ]
    with open(log_path, "w") as log_fh:
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=log_fh)
    try:
        _await_ready(proc, timeout=5.0)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        sock.connect(sock_path)
        time.sleep(0.1)
        sock.send(msgpack.packb(("__stop__",), use_bin_type=True))
        rc = proc.wait(timeout=5.0)
        assert rc == 0
        sock.close()
    finally:
        if proc.poll() is None:
            proc.kill()
        if os.path.exists(sock_path):
            try:
                os.unlink(sock_path)
            except OSError:
                pass
    samples = pl.read_parquet(tmp_path / "samples.parquet")
    assert "fake_energy_mj" in samples["domain"].unique().to_list()
