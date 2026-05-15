"""Direct pynvml backend for GPU utilization and per-PID VRAM.

This is independent of zeus_gpu's energy reads — both can run concurrently
without double-initializing NVML (nvmlInit is idempotent except for an
`AlreadyInitialized` flavor we treat as success).

Emits per GPU index `i`:
  - gpu{i}_util_pct        (util_pct; SM utilization)
  - gpu{i}_mem_util_pct    (util_pct; memory controller utilization)
  - gpu{i}_mem_used_bytes  (bytes; device-wide VRAM used)
  - gpu{i}_proc_vram_bytes (bytes; VRAM used by this PID and descendants)
"""

from __future__ import annotations

import logging
import os
from typing import Iterable

from pyscope.backends.base import Sample

log = logging.getLogger("pyscope.backends.nvml_util")


def _safe_nvml_init(pynvml) -> None:
    try:
        pynvml.nvmlInit()
    except Exception as exc:
        msg = str(exc)
        if "Already" in msg or "ALREADY_INITIALIZED" in msg:
            return
        raise


class NvmlUtilBackend:
    name = "nvml_util"

    def __init__(self, target_pid: int | None = None) -> None:
        import psutil
        import pynvml

        self._pynvml = pynvml
        self._psutil = psutil
        _safe_nvml_init(pynvml)
        self._handles = []
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            self._handles.append(pynvml.nvmlDeviceGetHandleByIndex(i))
        self._target_pid = int(target_pid) if target_pid is not None else os.getpid()
        try:
            self._self_proc = psutil.Process(self._target_pid)
        except psutil.NoSuchProcess:
            self._self_proc = None

    @classmethod
    def is_available(cls) -> bool:
        try:
            import pynvml  # noqa: F401
        except Exception:
            return False
        try:
            _safe_nvml_init(pynvml)
            return pynvml.nvmlDeviceGetCount() > 0
        except Exception:
            return False

    def _pids_of_interest(self) -> set[int]:
        pids = {self._target_pid}
        if self._self_proc is None:
            return pids
        try:
            for child in self._self_proc.children(recursive=True):
                pids.add(child.pid)
        except Exception:
            pass
        return pids

    def read(self) -> Iterable[Sample]:
        pynvml = self._pynvml
        out: list[Sample] = []
        pids_of_interest = self._pids_of_interest()

        for i, h in enumerate(self._handles):
            try:
                u = pynvml.nvmlDeviceGetUtilizationRates(h)
                out.append((f"gpu{i}_util_pct", float(u.gpu), "util_pct"))
                out.append((f"gpu{i}_mem_util_pct", float(u.memory), "util_pct"))
            except Exception:
                log.debug("nvmlDeviceGetUtilizationRates failed for gpu%d", i, exc_info=True)
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                out.append((f"gpu{i}_mem_used_bytes", float(mem.used), "bytes"))
            except Exception:
                log.debug("nvmlDeviceGetMemoryInfo failed for gpu%d", i, exc_info=True)
            try:
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
            except Exception:
                procs = []
            total = 0
            for p in procs:
                if p.pid in pids_of_interest and getattr(p, "usedGpuMemory", None):
                    total += int(p.usedGpuMemory)
            out.append((f"gpu{i}_proc_vram_bytes", float(total), "bytes"))

        return out

    def close(self) -> None:
        try:
            self._pynvml.nvmlShutdown()
        except Exception:
            pass
