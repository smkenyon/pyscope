"""Tests for the TDP fallback backend."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from pyscope.backends.tdp_fallback import TdpFallbackBackend, lookup_tdp_watts


def test_lookup_exact_xeon_platinum():
    w, src = lookup_tdp_watts("Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz")
    assert src == "exact"
    assert w == 270.0


def test_lookup_family_xeon_fallback():
    w, src = lookup_tdp_watts("Intel(R) Xeon(R) Unknown Model 12345")
    assert src == "family:intel_xeon"
    assert w > 0


def test_lookup_family_epyc():
    w, src = lookup_tdp_watts("AMD EPYC 9999 Processor")
    assert src == "family:amd_epyc"


def test_lookup_graviton_neoverse():
    w, src = lookup_tdp_watts("Neoverse-N1")
    assert src == "exact"
    assert w == 100.0


def test_lookup_generic_fallback():
    w, src = lookup_tdp_watts("Unknown CPU 9000")
    assert src == "family:generic"
    assert w > 0


def test_backend_emits_power_estimated():
    b = TdpFallbackBackend()
    samples = list(b.read())
    assert len(samples) == 1
    domain, value, kind = samples[0]
    assert domain == "cpu_total_power_mw_est"
    assert kind == "power_mw_estimated"
    assert value >= 0.0


def test_backend_power_proportional_to_util():
    """With util mocked, output power = TDP_W * 1000 * (util/100) mW."""
    b = TdpFallbackBackend()
    # Force a known TDP and util reading.
    b._tdp_w = 100.0
    with patch.object(b._psutil, "cpu_percent", return_value=50.0):
        _, value, _ = list(b.read())[0]
    assert value == pytest.approx(100.0 * 10.0 * 50.0)  # 50_000 mW = 50 W


def test_activation_skipped_when_zeus_cpu_active(monkeypatch):
    """tdp_fallback should not be auto-discovered when zeus_cpu is present."""
    from pyscope.monitor import _auto_discover_backends

    # Force zeus_cpu to look available, with a stub init.
    from pyscope.backends import zeus_cpu as zc_mod

    monkeypatch.setattr(zc_mod.ZeusCpuBackend, "is_available", classmethod(lambda cls: True))

    class StubZC(zc_mod.ZeusCpuBackend):
        def __init__(self):  # avoid touching real zeus
            self._cpus = type("_", (), {"cpus": []})()
            self._supports_dram = {}

    monkeypatch.setattr(zc_mod, "ZeusCpuBackend", StubZC)
    backends = _auto_discover_backends()
    names = {b.name for b in backends}
    assert "zeus_cpu" in names
    assert "tdp_fallback" not in names


def test_activation_happens_when_zeus_cpu_unavailable(monkeypatch):
    from pyscope.backends import zeus_cpu as zc_mod
    from pyscope.monitor import _auto_discover_backends

    monkeypatch.setattr(zc_mod.ZeusCpuBackend, "is_available", classmethod(lambda cls: False))
    backends = _auto_discover_backends()
    names = {b.name for b in backends}
    assert "tdp_fallback" in names
