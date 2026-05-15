"""zeus.device.get_cpus()-backed CPU energy reader.

Zeus's CPU layer (RAPL on Linux) returns a `CpuDramMeasurement(cpu_mj, dram_mj)`
per socket. Zeus handles counter wrap internally — we treat its readings as
already-monotonic energy in millijoules.

Emits per socket index `i`:
  - cpu{i}_energy_mj  (kind=energy_mj)
  - cpu{i}_dram_energy_mj  (kind=energy_mj; only if supported)

is_available is False when RAPL sysfs is missing or unreadable (typical
inside unprivileged containers). The Monitor logs the failure once and
continues without this backend.
"""

from __future__ import annotations

import logging
import os
from typing import Iterable

from pyscope.backends.base import Sample

log = logging.getLogger("pyscope.backends.zeus_cpu")

RAPL_SYSFS = "/sys/class/powercap/intel-rapl"


class ZeusCpuBackend:
    name = "zeus_cpu"

    def __init__(self) -> None:
        from zeus.device import get_cpus

        self._cpus = get_cpus()
        # cache supports_dram lookups per socket
        self._supports_dram: dict[int, bool] = {}
        for i, cpu in enumerate(self._cpus.cpus):
            try:
                self._supports_dram[i] = bool(cpu.supports_get_dram_energy_consumption())
            except Exception:
                self._supports_dram[i] = False

    @classmethod
    def is_available(cls) -> bool:
        # Fast path: missing sysfs tree.
        if not os.path.isdir(RAPL_SYSFS):
            return False
        # Verify zeus agrees and that we can read at least one energy zone.
        try:
            from zeus.device.cpu import rapl_is_available

            if not rapl_is_available():
                return False
        except Exception:
            log.exception("zeus.device.cpu import failed")
            return False
        # Permission probe: try to open any energy_uj under intel-rapl.
        try:
            for entry in os.listdir(RAPL_SYSFS):
                path = os.path.join(RAPL_SYSFS, entry, "energy_uj")
                if os.path.isfile(path):
                    with open(path) as fh:
                        fh.read(1)
                    return True
        except PermissionError:
            log.warning(
                "RAPL sysfs is present but not readable (%s); "
                "run with sudo or use zeusd to grant access",
                RAPL_SYSFS,
            )
            return False
        except OSError:
            return False
        return False

    def read(self) -> Iterable[Sample]:
        out: list[Sample] = []
        for i, cpu in enumerate(self._cpus.cpus):
            try:
                m = cpu.get_total_energy_consumption()
            except Exception:
                log.exception("zeus cpu[%d].get_total_energy_consumption failed", i)
                continue
            out.append((f"cpu{i}_energy_mj", float(m.cpu_mj), "energy_mj"))
            if m.dram_mj is not None and self._supports_dram.get(i, False):
                out.append((f"cpu{i}_dram_energy_mj", float(m.dram_mj), "energy_mj"))
        return out

    def close(self) -> None:
        return None
