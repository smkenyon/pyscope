"""perf fanout stub.

Writes pyscope labels to a FIFO at `/tmp/pyscope-perf.fifo` (if it exists)
so a `perf script` consumer can correlate trace points. This is a v1 stub —
the FIFO must be created by the user; we never create or destroy it.
"""

from __future__ import annotations

import os

PERF_FIFO_PATH = "/tmp/pyscope-perf.fifo"


class PerfFanout:
    name = "perf"

    def __init__(self) -> None:
        self._fd: int | None = None
        if os.path.exists(PERF_FIFO_PATH):
            try:
                self._fd = os.open(PERF_FIFO_PATH, os.O_WRONLY | os.O_NONBLOCK)
            except OSError:
                self._fd = None

    @classmethod
    def is_available(cls) -> bool:
        # Always "available" as a no-op when the FIFO is absent.
        return True

    def _write(self, line: str) -> None:
        if self._fd is None:
            return
        try:
            os.write(self._fd, (line + "\n").encode("utf-8", errors="replace"))
        except OSError:
            # Reader gone or pipe full — give up silently.
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

    def on_point(self, label: str, metadata: dict) -> None:
        self._write(f"point\t{label}")

    def on_enter(self, label: str, metadata: dict) -> None:
        self._write(f"enter\t{label}")

    def on_exit(self, label: str, metadata: dict) -> None:
        self._write(f"exit\t{label}")
