"""zeus.device.get_soc()-backed SoC reader for Apple Silicon and Jetson.

`SoCMeasurement` is a dataclass with platform-specific fields (Apple:
efficiency/performance cores, GPU, ANE; Jetson: CPU/GPU/SoC rails). We
introspect fields at read time and emit each numeric, non-None field as
`soc_{field}_energy_mj`. Energy is already in millijoules per zeus's
contract.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Iterable

from pyscope.backends.base import Sample

log = logging.getLogger("pyscope.backends.zeus_soc")


class ZeusSocBackend:
    name = "zeus_soc"

    def __init__(self) -> None:
        from zeus.device import get_soc

        self._soc = get_soc()
        try:
            self._available_metrics = set(self._soc.get_available_metrics())
        except Exception:
            self._available_metrics = set()
        log.info("zeus_soc available metrics: %s", self._available_metrics)

    @classmethod
    def is_available(cls) -> bool:
        try:
            from zeus.device.soc import (
                apple_silicon_is_available,
                jetson_is_available,
            )
        except Exception:
            return False
        return bool(apple_silicon_is_available() or jetson_is_available())

    def read(self) -> Iterable[Sample]:
        try:
            m = self._soc.get_total_energy_consumption()
        except Exception:
            log.exception("zeus soc.get_total_energy_consumption failed")
            return []
        if not dataclasses.is_dataclass(m):
            return []
        out: list[Sample] = []
        for f in dataclasses.fields(m):
            val = getattr(m, f.name, None)
            if val is None:
                continue
            try:
                v = float(val)
            except (TypeError, ValueError):
                continue
            out.append((f"soc_{f.name}_energy_mj", v, "energy_mj"))
        return out

    def close(self) -> None:
        return None
