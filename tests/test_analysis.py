"""AnalysisResult aggregation tests with synthetic samples and events."""

from __future__ import annotations

from pyscope.analysis import (
    build_analysis,
    energy_in_window,
    util_in_window,
)
from pyscope.events import Event


def _evt(ts, label, role, tid=1, **md):
    return Event(ts_ns=ts, label=label, role=role, metadata=md, thread_id=tid)


def _sample(ts, source, domain, value, kind):
    return (ts, source, domain, value, kind)


def test_energy_window_uses_last_minus_first():
    samples = [
        _sample(100, "zeus_cpu", "cpu0_energy_mj", 100.0, "energy_mj"),
        _sample(200, "zeus_cpu", "cpu0_energy_mj", 350.0, "energy_mj"),
        _sample(300, "zeus_cpu", "cpu0_energy_mj", 700.0, "energy_mj"),
    ]
    result = build_analysis(samples, events=[])
    e = energy_in_window(result.samples, 100, 300)
    assert e["cpu0_energy_mj"] == 600.0


def test_power_window_trapezoidal_integral_to_derived_mj():
    # Constant 1000 mW over 2 s (2e9 ns) → 2000 mJ.
    samples = [
        _sample(0, "zeus_gpu", "gpu0_power_mw", 1000.0, "power_mw"),
        _sample(1_000_000_000, "zeus_gpu", "gpu0_power_mw", 1000.0, "power_mw"),
        _sample(2_000_000_000, "zeus_gpu", "gpu0_power_mw", 1000.0, "power_mw"),
    ]
    result = build_analysis(samples, events=[])
    e = energy_in_window(result.samples, 0, 2_000_000_000)
    assert e["gpu0_power_mw::derived_mj"] == 2000.0


def test_util_window_stats():
    samples = [
        _sample(i, "nvml_util", "gpu0_util_pct", float(v), "util_pct")
        for i, v in enumerate([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ]
    result = build_analysis(samples, events=[])
    stats = util_in_window(result.samples, 0, 100)
    row = stats.row(0, named=True)
    assert row["domain"] == "gpu0_util_pct"
    assert row["max"] == 100.0
    assert row["mean"] == 55.0


def test_summary_segment_per_label():
    samples = [
        _sample(50, "zeus_cpu", "cpu0_energy_mj", 1000.0, "energy_mj"),
        _sample(150, "zeus_cpu", "cpu0_energy_mj", 1500.0, "energy_mj"),
        _sample(250, "zeus_cpu", "cpu0_energy_mj", 1900.0, "energy_mj"),
        _sample(100, "nvml_util", "gpu0_util_pct", 80.0, "util_pct"),
        _sample(150, "nvml_util", "gpu0_util_pct", 90.0, "util_pct"),
        _sample(200, "nvml_util", "gpu0_util_pct", 95.0, "util_pct"),
    ]
    events = [
        _evt(100, "train_step", "enter"),
        _evt(200, "train_step", "exit"),
    ]
    result = build_analysis(samples, events)
    summary = result.summary_frame()
    assert summary.height == 1
    row = summary.row(0, named=True)
    assert row["label"] == "train_step"
    # energy_mj last - first within [100, 200] = 1900-1000? No: samples
    # at ts 50, 150, 250. In window [100,200]: only ts=150. last-first=0.
    assert row["cpu_energy_J"] == 0.0
    # util p95 of [80, 90, 95]
    assert row["gpu_util_p95"] is not None
    assert row["source_quality"] == "measured"


def test_summary_source_quality_flag_when_estimated():
    samples = [
        _sample(0, "tdp_fallback", "cpu_total_power_mw_est", 50000.0, "power_mw_estimated"),
        _sample(1_000_000_000, "tdp_fallback", "cpu_total_power_mw_est", 50000.0, "power_mw_estimated"),
    ]
    events = [
        _evt(0, "x", "enter"),
        _evt(1_000_000_000, "x", "exit"),
    ]
    result = build_analysis(samples, events)
    row = result.summary_frame().row(0, named=True)
    assert row["source_quality"] == "estimated"
    # 50 W for 1 s = 50 J
    assert row["cpu_energy_J"] == 50.0


def test_summary_empty_when_no_events():
    result = build_analysis([], [])
    assert result.summary_frame().height == 0


def test_summary_handles_zero_duration_point():
    events = [_evt(42, "marker", "point")]
    result = build_analysis([], events)
    s = result.summary_frame()
    assert s.height == 1
    assert s.row(0, named=True)["duration_ms"] == 0.0


def test_analysis_result_helpers():
    samples = [
        _sample(100, "zeus_cpu", "cpu0_energy_mj", 100.0, "energy_mj"),
        _sample(200, "zeus_cpu", "cpu0_energy_mj", 500.0, "energy_mj"),
    ]
    events = [_evt(100, "blk", "enter"), _evt(200, "blk", "exit")]
    result = build_analysis(samples, events)
    assert result.energy_in_segment("blk")["cpu0_energy_mj"] == 400.0
    assert result.energy_in_segment("nonexistent") == {}


def test_write_parquet_round_trip(tmp_path):
    import polars as pl
    samples = [_sample(10, "fake", "fake_energy_mj", 1.0, "energy_mj")]
    events = [_evt(10, "p", "point")]
    result = build_analysis(samples, events)
    result.write_parquet(tmp_path)
    assert (tmp_path / "samples.parquet").exists()
    assert (tmp_path / "events.parquet").exists()
    assert (tmp_path / "segments.parquet").exists()
    s = pl.read_parquet(tmp_path / "samples.parquet")
    assert s.height == 1


def test_summary_table_format_produces_text():
    samples = [_sample(0, "fake", "fake_energy_mj", 1.0, "energy_mj")]
    events = [_evt(0, "x", "enter"), _evt(10, "x", "exit")]
    result = build_analysis(samples, events)
    text = result.summary("table")
    assert "x" in text  # label appears
    assert "pyscope summary" in text


def test_summary_json_format_parses():
    import json
    samples = [_sample(0, "fake", "fake_energy_mj", 1.0, "energy_mj")]
    events = [_evt(0, "x", "enter"), _evt(10, "x", "exit")]
    result = build_analysis(samples, events)
    j = json.loads(result.summary("json"))
    assert any(row["label"] == "x" for row in j)
