"""Tests for fanout emitters: enter/exit pairing, exception safety, no-op when absent."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def fake_nvtx(monkeypatch):
    mod = types.ModuleType("nvtx")
    mod.mark = MagicMock()
    mod.range_push = MagicMock()
    mod.range_pop = MagicMock()
    monkeypatch.setitem(sys.modules, "nvtx", mod)
    yield mod


def test_nvtx_fanout_point_and_range(fake_nvtx):
    from pyscope.fanout.nvtx_out import NvtxFanout

    fo = NvtxFanout()
    fo.on_point("p", {})
    fake_nvtx.mark.assert_called_with(message="p")
    fo.on_enter("blk", {"x": 1})
    fake_nvtx.range_push.assert_called_with(message="blk")
    fo.on_exit("blk", {})
    fake_nvtx.range_pop.assert_called_once()


def test_nvtx_fanout_exit_runs_on_exception_via_monitor(fake_nvtx):
    """Monitor.scope catches exceptions and still calls on_exit."""
    from pyscope.backends._fake import FakeBackend
    from pyscope.fanout.nvtx_out import NvtxFanout
    from pyscope.monitor import Monitor, _reset_singleton_for_tests

    _reset_singleton_for_tests()
    m = Monitor(interval_ms=50, backends=[FakeBackend()], fanout=[NvtxFanout()])
    m.start()
    with pytest.raises(RuntimeError):
        with m.scope("boom"):
            raise RuntimeError("expected")
    m.stop()
    fake_nvtx.range_push.assert_called_with(message="boom")
    fake_nvtx.range_pop.assert_called_once()


@pytest.fixture
def fake_otel(monkeypatch):
    trace_mod = types.ModuleType("opentelemetry.trace")

    active_span = MagicMock()
    active_span.is_recording.return_value = True
    trace_mod.get_current_span = MagicMock(return_value=active_span)

    tracer = MagicMock()
    cm_obj = MagicMock()
    inner_span = MagicMock()
    cm_obj.__enter__ = MagicMock(return_value=inner_span)
    cm_obj.__exit__ = MagicMock(return_value=False)
    tracer.start_as_current_span = MagicMock(return_value=cm_obj)
    trace_mod.get_tracer = MagicMock(return_value=tracer)

    monkeypatch.setitem(sys.modules, "opentelemetry", types.ModuleType("opentelemetry"))
    monkeypatch.setitem(sys.modules, "opentelemetry.trace", trace_mod)
    yield trace_mod, active_span, tracer, cm_obj


def test_otel_fanout_point_uses_active_span(fake_otel):
    trace_mod, active_span, _tracer, _cm = fake_otel
    from pyscope.fanout.otel_out import OtelFanout

    fo = OtelFanout()
    fo.on_point("evt", {"k": "v"})
    active_span.add_event.assert_called_with("evt", attributes={"k": "v"})


def test_otel_fanout_point_noop_without_active_span(fake_otel, monkeypatch):
    trace_mod, active_span, _tracer, _cm = fake_otel
    active_span.is_recording.return_value = False
    from pyscope.fanout.otel_out import OtelFanout

    fo = OtelFanout()
    fo.on_point("evt", {})
    active_span.add_event.assert_not_called()


def test_otel_fanout_scope_starts_and_ends_span(fake_otel):
    trace_mod, _active, tracer, cm = fake_otel
    from pyscope.fanout.otel_out import OtelFanout

    fo = OtelFanout()
    fo.on_enter("blk", {"id": 1})
    tracer.start_as_current_span.assert_called_with("blk", attributes={"id": 1})
    cm.__enter__.assert_called()
    fo.on_exit("blk", {})
    cm.__exit__.assert_called()


def test_perf_fanout_is_noop_without_fifo(tmp_path, monkeypatch):
    monkeypatch.setattr("pyscope.fanout.perf_out.PERF_FIFO_PATH", str(tmp_path / "missing.fifo"))
    from pyscope.fanout.perf_out import PerfFanout

    fo = PerfFanout()
    # Should not raise.
    fo.on_point("p", {})
    fo.on_enter("blk", {})
    fo.on_exit("blk", {})


def test_perf_fanout_writes_when_fifo_open(tmp_path, monkeypatch):
    import os

    fifo = tmp_path / "p.fifo"
    os.mkfifo(fifo)
    monkeypatch.setattr("pyscope.fanout.perf_out.PERF_FIFO_PATH", str(fifo))

    # Open a reader so the write side won't block / error.
    rfd = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
    try:
        from pyscope.fanout.perf_out import PerfFanout

        fo = PerfFanout()
        fo.on_point("hello", {})
        fo.on_enter("x", {})
        fo.on_exit("x", {})
        # Read what was written.
        data = b""
        try:
            data = os.read(rfd, 4096)
        except BlockingIOError:
            pass
        text = data.decode()
        assert "point\thello" in text
        assert "enter\tx" in text
        assert "exit\tx" in text
    finally:
        os.close(rfd)


def test_auto_discover_fanout_includes_nvtx_when_present(fake_nvtx):
    from pyscope.monitor import _auto_discover_fanout

    fanouts = _auto_discover_fanout()
    names = {getattr(f, "name", "") for f in fanouts}
    assert "nvtx" in names
