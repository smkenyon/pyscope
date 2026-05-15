"""Integration test for the psutil_sys backend."""

from __future__ import annotations

import time

import pytest

from pyscope.backends.psutil_sys import PsutilSysBackend
from pyscope.monitor import Monitor, _reset_singleton_for_tests


@pytest.fixture(autouse=True)
def _reset():
    _reset_singleton_for_tests()
    yield
    _reset_singleton_for_tests()


def test_psutil_is_available():
    assert PsutilSysBackend.is_available()


def test_psutil_backend_emits_expected_domains():
    b = PsutilSysBackend()
    samples = list(b.read())
    domains = {d for d, _, _ in samples}
    assert "system_ram_used_bytes" in domains
    assert "system_ram_available_bytes" in domains
    assert "cpu_total_util_pct" in domains
    assert any(d.startswith("cpu") and d.endswith("_util_pct") and d != "cpu_total_util_pct" for d in domains)
    b.close()


def test_psutil_under_monitor_collects_samples():
    m = Monitor(interval_ms=30, backends=[PsutilSysBackend()])
    m.start()
    time.sleep(0.3)
    m.stop()
    result = m.analyze()
    domains = set(result.samples["domain"].unique().to_list())
    assert "system_ram_used_bytes" in domains
    assert "cpu_total_util_pct" in domains
    ram_rows = result.samples.filter(result.samples["domain"] == "system_ram_used_bytes")
    assert (ram_rows["value"] > 0).all()
