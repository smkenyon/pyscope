"""zeus.device.get_gpus()-backed GPU energy/power reader.

Prefers `get_total_energy_consumption()` (Volta+ on NVIDIA, supported AMD GPUs).
For older GPUs that don't support cumulative energy, falls back to
`get_instant_power_usage()` and emits `power_mw`. Analysis integrates power
samples into derived energy.

Emits per GPU index `i`:
  - gpu{i}_energy_mj  (kind=energy_mj)   when supported
  - gpu{i}_power_mw   (kind=power_mw)    when energy not supported

`is_available` is False when no GPU vendor backend is loadable.
"""

from __future__ import annotations

import logging
from typing import Iterable

from pyscope.backends.base import Sample

log = logging.getLogger("pyscope.backends.zeus_gpu")


class ZeusGpuBackend:
    name = "zeus_gpu"

    def __init__(self) -> None:
        from zeus.device import get_gpus

        self._gpus = get_gpus()
        self._supports_energy: dict[int, bool] = {}
        for i, gpu in enumerate(self._gpus.gpus):
            try:
                self._supports_energy[i] = bool(gpu.supports_get_total_energy_consumption())
            except Exception:
                self._supports_energy[i] = False

    @classmethod
    def is_available(cls) -> bool:
        try:
            from zeus.device.gpu import amdsmi_is_available, nvml_is_available
        except Exception:
            return False
        return bool(nvml_is_available() or amdsmi_is_available())

    def read(self) -> Iterable[Sample]:
        out: list[Sample] = []
        for i, gpu in enumerate(self._gpus.gpus):
            if self._supports_energy.get(i, False):
                try:
                    mj = gpu.get_total_energy_consumption()
                    out.append((f"gpu{i}_energy_mj", float(mj), "energy_mj"))
                    continue
                except Exception:
                    log.exception("zeus gpu[%d] energy read failed; falling back to power", i)
            # Power fallback.
            try:
                mw = gpu.get_instant_power_usage()
                out.append((f"gpu{i}_power_mw", float(mw), "power_mw"))
            except Exception:
                log.exception("zeus gpu[%d] power read failed", i)
        return out

    def close(self) -> None:
        return None
