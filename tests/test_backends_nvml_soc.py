"""Tests for NvmlUtilBackend and ZeusSocBackend using mocks."""

from __future__ import annotations

import dataclasses
import os
import sys
import types
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def fake_pynvml(monkeypatch):
    mod = types.ModuleType("pynvml")
    mod.nvmlInit = MagicMock()
    mod.nvmlShutdown = MagicMock()
    mod.nvmlDeviceGetCount = MagicMock(return_value=1)
    handle = MagicMock(name="gpu0")
    mod.nvmlDeviceGetHandleByIndex = MagicMock(return_value=handle)

    util = MagicMock(gpu=42, memory=17)
    mod.nvmlDeviceGetUtilizationRates = MagicMock(return_value=util)

    mem = MagicMock(used=1024 * 1024 * 500)
    mod.nvmlDeviceGetMemoryInfo = MagicMock(return_value=mem)

    my_pid = os.getpid()
    proc_self = MagicMock(pid=my_pid, usedGpuMemory=1024 * 1024 * 200)
    proc_other = MagicMock(pid=999999, usedGpuMemory=1024 * 1024 * 800)
    mod.nvmlDeviceGetComputeRunningProcesses = MagicMock(return_value=[proc_self, proc_other])

    monkeypatch.setitem(sys.modules, "pynvml", mod)
    yield mod


def test_nvml_util_emits_expected_domains(fake_pynvml):
    from pyscope.backends.nvml_util import NvmlUtilBackend

    b = NvmlUtilBackend()
    samples = list(b.read())
    domains = {d: (v, k) for d, v, k in samples}
    assert domains["gpu0_util_pct"] == (42.0, "util_pct")
    assert domains["gpu0_mem_util_pct"] == (17.0, "util_pct")
    assert domains["gpu0_mem_used_bytes"][1] == "bytes"
    # Only this PID's VRAM counted, not the other process.
    assert domains["gpu0_proc_vram_bytes"] == (float(1024 * 1024 * 200), "bytes")


def test_nvml_util_is_available_no_devices(fake_pynvml):
    fake_pynvml.nvmlDeviceGetCount = MagicMock(return_value=0)
    from pyscope.backends.nvml_util import NvmlUtilBackend

    assert NvmlUtilBackend.is_available() is False


def test_nvml_util_safe_init_handles_already_initialized(fake_pynvml):
    def raise_already(*a, **kw):
        raise RuntimeError("NVML_ERROR_ALREADY_INITIALIZED")

    fake_pynvml.nvmlInit = MagicMock(side_effect=raise_already)
    # Should not raise.
    from pyscope.backends.nvml_util import NvmlUtilBackend

    b = NvmlUtilBackend()
    assert b is not None
    b.close()


def test_zeus_soc_emits_per_field(monkeypatch):
    @dataclasses.dataclass
    class FakeSoCMeasurement:
        cpu_mj: float = 100.0
        gpu_mj: float | None = 200.0
        ane_mj: float | None = None

    fake_soc_obj = MagicMock()
    fake_soc_obj.get_available_metrics.return_value = {"cpu_mj", "gpu_mj", "ane_mj"}
    fake_soc_obj.get_total_energy_consumption.return_value = FakeSoCMeasurement()

    fake_device = types.ModuleType("zeus.device")
    fake_device.get_soc = lambda: fake_soc_obj
    fake_device.get_cpus = MagicMock()
    fake_device.get_gpus = MagicMock()
    monkeypatch.setitem(sys.modules, "zeus.device", fake_device)

    fake_soc_mod = types.ModuleType("zeus.device.soc")
    fake_soc_mod.apple_silicon_is_available = lambda: True
    fake_soc_mod.jetson_is_available = lambda: False
    monkeypatch.setitem(sys.modules, "zeus.device.soc", fake_soc_mod)

    from pyscope.backends.zeus_soc import ZeusSocBackend

    assert ZeusSocBackend.is_available()
    b = ZeusSocBackend()
    samples = list(b.read())
    domains = {d: v for d, v, _ in samples}
    assert domains["soc_cpu_mj_energy_mj"] == 100.0
    assert domains["soc_gpu_mj_energy_mj"] == 200.0
    # None-valued fields are skipped.
    assert "soc_ane_mj_energy_mj" not in domains


def test_zeus_soc_unavailable_when_not_apple_or_jetson(monkeypatch):
    fake_soc_mod = types.ModuleType("zeus.device.soc")
    fake_soc_mod.apple_silicon_is_available = lambda: False
    fake_soc_mod.jetson_is_available = lambda: False
    monkeypatch.setitem(sys.modules, "zeus.device.soc", fake_soc_mod)
    from pyscope.backends.zeus_soc import ZeusSocBackend

    assert ZeusSocBackend.is_available() is False
