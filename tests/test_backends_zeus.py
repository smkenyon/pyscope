"""Tests for ZeusCpuBackend and ZeusGpuBackend using mocked zeus.device.

These tests do not require a host with RAPL or GPU access.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def fake_zeus(monkeypatch):
    """Inject a fake zeus.device that the backends can import."""
    cpu_obj = MagicMock()
    cpu_obj.supports_get_dram_energy_consumption.return_value = True
    cpu_meas = MagicMock(cpu_mj=12345.0, dram_mj=678.0)
    cpu_obj.get_total_energy_consumption.return_value = cpu_meas

    cpus = MagicMock()
    cpus.cpus = [cpu_obj]

    gpu_obj_energy = MagicMock()
    gpu_obj_energy.supports_get_total_energy_consumption.return_value = True
    gpu_obj_energy.get_total_energy_consumption.return_value = 50000

    gpu_obj_power = MagicMock()
    gpu_obj_power.supports_get_total_energy_consumption.return_value = False
    gpu_obj_power.get_instant_power_usage.return_value = 75000

    gpus = MagicMock()
    gpus.gpus = [gpu_obj_energy, gpu_obj_power]

    fake_device = types.ModuleType("zeus.device")
    fake_device.get_cpus = lambda: cpus
    fake_device.get_gpus = lambda: gpus
    fake_device.get_soc = lambda: MagicMock()

    fake_gpu_mod = types.ModuleType("zeus.device.gpu")
    fake_gpu_mod.nvml_is_available = lambda: True
    fake_gpu_mod.amdsmi_is_available = lambda: False

    fake_cpu_mod = types.ModuleType("zeus.device.cpu")
    fake_cpu_mod.rapl_is_available = lambda: True

    monkeypatch.setitem(sys.modules, "zeus.device", fake_device)
    monkeypatch.setitem(sys.modules, "zeus.device.gpu", fake_gpu_mod)
    monkeypatch.setitem(sys.modules, "zeus.device.cpu", fake_cpu_mod)
    yield cpus, gpus


def test_zeus_cpu_emits_energy_and_dram(fake_zeus):
    from pyscope.backends.zeus_cpu import ZeusCpuBackend

    b = ZeusCpuBackend()
    samples = list(b.read())
    domains = {d: (v, k) for d, v, k in samples}
    assert "cpu0_energy_mj" in domains
    assert domains["cpu0_energy_mj"] == (12345.0, "energy_mj")
    assert "cpu0_dram_energy_mj" in domains
    assert domains["cpu0_dram_energy_mj"] == (678.0, "energy_mj")


def test_zeus_cpu_no_dram_when_unsupported(fake_zeus, monkeypatch):
    cpus, _ = fake_zeus
    cpus.cpus[0].supports_get_dram_energy_consumption.return_value = False
    from pyscope.backends.zeus_cpu import ZeusCpuBackend

    b = ZeusCpuBackend()
    samples = list(b.read())
    domains = {d for d, _, _ in samples}
    assert "cpu0_energy_mj" in domains
    assert "cpu0_dram_energy_mj" not in domains


def test_zeus_cpu_is_available_without_rapl_sysfs():
    from pyscope.backends.zeus_cpu import ZeusCpuBackend

    with patch("os.path.isdir", return_value=False):
        assert ZeusCpuBackend.is_available() is False


def test_zeus_cpu_is_available_with_permission_error(monkeypatch):
    from pyscope.backends import zeus_cpu as mod

    monkeypatch.setattr("os.path.isdir", lambda p: True)
    fake_cpu_mod = types.ModuleType("zeus.device.cpu")
    fake_cpu_mod.rapl_is_available = lambda: True
    monkeypatch.setitem(sys.modules, "zeus.device.cpu", fake_cpu_mod)
    monkeypatch.setattr("os.listdir", lambda p: ["intel-rapl:0"])
    monkeypatch.setattr("os.path.isfile", lambda p: True)

    def boom(*a, **kw):
        raise PermissionError("nope")

    monkeypatch.setattr("builtins.open", boom)
    assert mod.ZeusCpuBackend.is_available() is False


def test_zeus_gpu_prefers_energy_when_supported(fake_zeus):
    from pyscope.backends.zeus_gpu import ZeusGpuBackend

    b = ZeusGpuBackend()
    samples = list(b.read())
    domains = {d: (v, k) for d, v, k in samples}
    assert domains["gpu0_energy_mj"] == (50000.0, "energy_mj")
    # gpu1 falls back to power.
    assert domains["gpu1_power_mw"] == (75000.0, "power_mw")
    assert "gpu1_energy_mj" not in domains


def test_zeus_gpu_falls_back_when_energy_raises(fake_zeus):
    cpus, gpus = fake_zeus
    gpus.gpus[0].get_total_energy_consumption.side_effect = RuntimeError("not supported on Pascal")
    gpus.gpus[0].get_instant_power_usage.return_value = 90000
    from pyscope.backends.zeus_gpu import ZeusGpuBackend

    b = ZeusGpuBackend()
    samples = list(b.read())
    domains = {d: v for d, v, _ in samples}
    assert domains.get("gpu0_power_mw") == 90000.0


def test_zeus_gpu_is_available_false_when_no_vendor(monkeypatch):
    fake_gpu_mod = types.ModuleType("zeus.device.gpu")
    fake_gpu_mod.nvml_is_available = lambda: False
    fake_gpu_mod.amdsmi_is_available = lambda: False
    monkeypatch.setitem(sys.modules, "zeus.device.gpu", fake_gpu_mod)
    from pyscope.backends.zeus_gpu import ZeusGpuBackend

    assert ZeusGpuBackend.is_available() is False
