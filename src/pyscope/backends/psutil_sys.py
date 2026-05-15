"""psutil-based system resource backend.

Emits, every tick:
  - system_ram_used_bytes / system_ram_available_bytes (bytes)
  - proc_tree_rss_bytes (bytes; this PID + descendants)
  - cpu_total_util_pct (util_pct; aggregate across all cores)
  - cpu{i}_util_pct (util_pct; per-core)

Disk and network I/O are intentionally off in v1 (low value vs. high noise).
"""

from __future__ import annotations

import logging
import os
from typing import Iterable

from pyscope.backends.base import Sample

log = logging.getLogger("pyscope.backends.psutil_sys")


class PsutilSysBackend:
    name = "psutil_sys"

    def __init__(self) -> None:
        import psutil  # local import; tested via is_available

        self._psutil = psutil
        self._self_proc = psutil.Process(os.getpid())
        # psutil.cpu_percent needs a priming call to set its baseline; the
        # first reading after start() is therefore meaningful.
        psutil.cpu_percent(interval=None, percpu=False)
        psutil.cpu_percent(interval=None, percpu=True)
        self._self_proc.cpu_percent(interval=None)

    @classmethod
    def is_available(cls) -> bool:
        try:
            import psutil  # noqa: F401
        except Exception:
            return False
        return True

    def read(self) -> Iterable[Sample]:
        psutil = self._psutil
        out: list[Sample] = []

        vm = psutil.virtual_memory()
        out.append(("system_ram_used_bytes", float(vm.used), "bytes"))
        out.append(("system_ram_available_bytes", float(vm.available), "bytes"))

        try:
            rss = self._self_proc.memory_info().rss
            for child in self._self_proc.children(recursive=True):
                try:
                    rss += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            out.append(("proc_tree_rss_bytes", float(rss), "bytes"))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        total = psutil.cpu_percent(interval=None, percpu=False)
        out.append(("cpu_total_util_pct", float(total), "util_pct"))
        for i, pct in enumerate(psutil.cpu_percent(interval=None, percpu=True)):
            out.append((f"cpu{i}_util_pct", float(pct), "util_pct"))

        return out

    def close(self) -> None:
        return None
