"""Tests for Monitor + Sampler integration using the FakeBackend."""

from __future__ import annotations

import threading
import time

import pytest

import pyscope
from pyscope.backends._fake import FakeBackend
from pyscope.monitor import Monitor, _reset_singleton_for_tests


@pytest.fixture(autouse=True)
def _reset_singleton():
    _reset_singleton_for_tests()
    yield
    _reset_singleton_for_tests()


def test_start_stop_collects_samples():
    m = Monitor(interval_ms=20, backends=[FakeBackend()])
    m.start()
    time.sleep(0.2)  # ~10 ticks
    m.stop()
    result = m.analyze()
    # Two domains per tick. Allow generous tolerance for jittery hosts.
    assert result.samples.height >= 8
    assert set(result.samples["domain"].unique().to_list()) == {"fake_energy_mj", "fake_util_pct"}
    ts = result.samples["ts_ns"].to_list()
    assert ts == sorted(ts)


def test_double_start_is_idempotent():
    m = Monitor(interval_ms=50, backends=[FakeBackend()])
    m.start()
    m.start()  # should warn, not raise
    assert m.is_running
    m.stop()


def test_stop_when_not_running_is_noop():
    m = Monitor(interval_ms=50, backends=[FakeBackend()])
    m.stop()  # no-op, no raise


def test_annotate_and_scope_record_events():
    m = Monitor(interval_ms=50, backends=[FakeBackend()])
    m.start()
    m.annotate("point-1", k=1)
    with m.scope("blk", batch=4):
        m.annotate("inside")
    m.stop()
    events = m.events.snapshot()
    roles = [e.role for e in events]
    labels = [e.label for e in events]
    assert roles == ["point", "enter", "point", "exit"]
    assert labels == ["point-1", "blk", "inside", "blk"]
    assert events[1].metadata == {"batch": 4}
    assert events[1].thread_id == threading.get_ident()


def test_scoped_decorator():
    m = Monitor(interval_ms=50, backends=[FakeBackend()])
    m.start()

    @m.scoped("decorated")
    def work(x: int) -> int:
        return x * 2

    assert work(21) == 42
    m.stop()
    events = m.events.snapshot()
    assert [e.role for e in events if e.label == "decorated"] == ["enter", "exit"]


def test_scoped_rejects_async():
    m = Monitor(interval_ms=50, backends=[FakeBackend()])

    async def coro() -> None:
        pass

    with pytest.raises(TypeError):
        m.scoped("nope")(coro)


def test_scope_exits_on_exception():
    m = Monitor(interval_ms=50, backends=[FakeBackend()])
    m.start()
    with pytest.raises(RuntimeError):
        with m.scope("oops"):
            raise RuntimeError("boom")
    m.stop()
    events = m.events.snapshot()
    roles = [e.role for e in events if e.label == "oops"]
    assert roles == ["enter", "exit"]


def test_module_singleton_start_stop():
    pyscope.start(interval_ms=20, backends=[FakeBackend()])
    pyscope.annotate("hi")
    with pyscope.scope("blk"):
        pass
    time.sleep(0.1)
    pyscope.stop()
    result = pyscope.analyze()
    labels = result.events.filter(
        (result.events["label"] == "blk") & (result.events["role"] == "enter")
    )
    assert labels.height == 1
    assert result.samples.height > 0
    assert result.wall_clock_anchor_ns > 0


def test_sampler_cadence_within_tolerance():
    """At 50 ms interval over 250 ms we expect ~5 ticks ±2."""
    m = Monitor(interval_ms=50, backends=[FakeBackend()])
    m.start()
    time.sleep(0.25)
    m.stop()
    result = m.analyze()
    # Two samples per tick.
    ticks = result.samples.height // 2
    assert 3 <= ticks <= 8, f"unexpected tick count {ticks}"


def test_backend_failure_isolation():
    class FlakyBackend:
        name = "flaky"

        @classmethod
        def is_available(cls):
            return True

        def __init__(self):
            self.calls = 0

        def read(self):
            self.calls += 1
            raise RuntimeError("nope")

        def close(self):
            pass

    flaky = FlakyBackend()
    m = Monitor(interval_ms=10, backends=[flaky, FakeBackend()])
    m.start()
    time.sleep(0.2)
    m.stop()
    result = m.analyze()
    # FakeBackend continues to produce data despite flaky failing.
    assert (result.samples["source"] == "fake").any()
    # Flaky should have been dropped after MAX_BACKEND_FAILS.
    assert flaky.calls >= 5
