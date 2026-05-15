"""TDP × utilization power estimator for hosts without real CPU energy counters.

Used when `zeus_cpu.is_available()` is False (rootless containers, ARM cloud
guests like Graviton, etc.). The estimate is necessarily crude: we multiply
a CPU TDP lookup (from a small bundled vendor/model table) by current CPU
utilization across all cores and emit `power_mw_estimated` so downstream
analysis can flag derived values.

Activation gating happens at auto-discovery time in monitor.py, not here —
this backend will run if instantiated unconditionally.
"""

from __future__ import annotations

import logging
import platform
import re
from typing import Iterable

from pyscope.backends.base import Sample

log = logging.getLogger("pyscope.backends.tdp_fallback")

# Conservative defaults by vendor family. Real-world TDPs vary widely; the
# table prefers under-estimation to over-estimation. Numbers in watts.
_VENDOR_DEFAULTS: dict[str, float] = {
    "intel_xeon": 150.0,
    "intel_core": 65.0,
    "amd_epyc": 200.0,
    "amd_ryzen": 95.0,
    "graviton": 100.0,
    "ampere": 130.0,
    "apple_m": 20.0,
    "generic": 65.0,
}

# Specific model overrides — populate sparingly. Keys are normalized to
# lowercase with collapsed whitespace.
_MODEL_TABLE: dict[str, float] = {
    # examples; extend as needed
    "intel(r) xeon(r) platinum 8375c cpu @ 2.90ghz": 270.0,
    "intel(r) xeon(r) cpu e5-2680 v4 @ 2.40ghz": 120.0,
    "amd epyc 7r32 48-core processor": 280.0,
    "neoverse-n1": 100.0,  # Graviton2
    "neoverse-v1": 130.0,  # Graviton3
}


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _detect_cpu_model() -> str:
    # Prefer /proc/cpuinfo on Linux — platform.processor() is often empty.
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.lower().startswith(("model name", "cpu part", "cpu implementer")):
                    _, _, val = line.partition(":")
                    val = val.strip()
                    if val:
                        return val
    except OSError:
        pass
    return platform.processor() or ""


def lookup_tdp_watts(model: str) -> tuple[float, str]:
    """Return (watts, source_tag). `source_tag` indicates whether this came
    from an exact model match or a vendor-family default.
    """
    key = _normalize(model)
    if key in _MODEL_TABLE:
        return _MODEL_TABLE[key], "exact"
    # Vendor heuristics.
    if "xeon" in key:
        return _VENDOR_DEFAULTS["intel_xeon"], "family:intel_xeon"
    if "core(tm) i" in key or "intel(r) core" in key:
        return _VENDOR_DEFAULTS["intel_core"], "family:intel_core"
    if "epyc" in key:
        return _VENDOR_DEFAULTS["amd_epyc"], "family:amd_epyc"
    if "ryzen" in key:
        return _VENDOR_DEFAULTS["amd_ryzen"], "family:amd_ryzen"
    if "graviton" in key or "neoverse" in key:
        return _VENDOR_DEFAULTS["graviton"], "family:graviton"
    if "ampere" in key:
        return _VENDOR_DEFAULTS["ampere"], "family:ampere"
    if "apple m" in key:
        return _VENDOR_DEFAULTS["apple_m"], "family:apple_m"
    return _VENDOR_DEFAULTS["generic"], "family:generic"


class TdpFallbackBackend:
    name = "tdp_fallback"

    def __init__(self) -> None:
        import psutil

        self._psutil = psutil
        self._model = _detect_cpu_model()
        self._tdp_w, self._source = lookup_tdp_watts(self._model)
        log.info(
            "tdp_fallback using TDP=%.1f W for model %r (source=%s)",
            self._tdp_w,
            self._model,
            self._source,
        )
        # Prime cpu_percent baseline.
        psutil.cpu_percent(interval=None, percpu=False)

    @classmethod
    def is_available(cls) -> bool:
        try:
            import psutil  # noqa: F401
        except Exception:
            return False
        return True

    def read(self) -> Iterable[Sample]:
        util = float(self._psutil.cpu_percent(interval=None, percpu=False))
        # Power in mW = TDP_W * 1000 * (util / 100)
        power_mw = self._tdp_w * 10.0 * util
        return [("cpu_total_power_mw_est", power_mw, "power_mw_estimated")]

    def close(self) -> None:
        return None
