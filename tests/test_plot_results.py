"""Smoke tests for the plot_results.py utility.

We don't compare PNG bytes; we just check the file is written and is non-
trivially sized. Matching axis labels would lock us in too tightly.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import polars as pl

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "plot_results.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("plot_results", SCRIPT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["plot_results"] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_parquet(tmp_path: Path) -> Path:
    """Write a small samples + events bundle covering every kind."""
    out = tmp_path / "out"
    out.mkdir()
    samples = pl.DataFrame(
        {
            "ts_ns": [
                0, 100_000_000, 200_000_000, 300_000_000,         # 4 ticks energy
                0, 100_000_000, 200_000_000, 300_000_000,         # 4 ticks util
                0, 100_000_000, 200_000_000, 300_000_000,         # 4 ticks ram
                0, 100_000_000, 200_000_000, 300_000_000,         # 4 ticks power
            ],
            "source": ["s"] * 16,
            "domain": (
                ["cpu0_energy_mj"] * 4
                + ["gpu0_util_pct"] * 4
                + ["system_ram_used_bytes"] * 4
                + ["gpu0_power_mw"] * 4
            ),
            "value": [
                100.0, 250.0, 400.0, 550.0,
                10.0, 80.0, 90.0, 50.0,
                1e9, 1.1e9, 1.05e9, 1.0e9,
                30000.0, 50000.0, 70000.0, 40000.0,
            ],
            "kind": (
                ["energy_mj"] * 4
                + ["util_pct"] * 4
                + ["bytes"] * 4
                + ["power_mw"] * 4
            ),
        }
    )
    events = pl.DataFrame(
        {
            "ts_ns": [50_000_000, 150_000_000, 220_000_000, 280_000_000],
            "label": ["preprocess", "inference", "inference", "postprocess"],
            "role": ["enter", "enter", "exit", "point"],
            "metadata": ["{}"] * 4,
            "thread_id": [1] * 4,
        }
    )
    samples.write_parquet(out / "samples.parquet")
    events.write_parquet(out / "events.parquet")
    return out


def test_main_writes_one_plot_per_domain(tmp_path):
    plot_results = _load_module()
    out = _write_parquet(tmp_path)

    rc = plot_results.main([str(out)])
    assert rc == 0
    plots = list((out / "plots").glob("*.png"))
    fnames = {p.name for p in plots}
    assert "cpu0_energy_mj.png" in fnames
    assert "gpu0_util_pct.png" in fnames
    assert "system_ram_used_bytes.png" in fnames
    assert "gpu0_power_mw.png" in fnames
    # All non-empty.
    for p in plots:
        assert p.stat().st_size > 1000


def test_annotation_subset_filter(tmp_path):
    plot_results = _load_module()
    out = _write_parquet(tmp_path)

    # Resolve annotations programmatically to assert filter semantics.
    events = pl.read_parquet(out / "events.parquet")
    full = plot_results._resolve_annotations(events, None)
    assert full.height == 4

    # Order should not matter — passing labels in any sequence picks the
    # same events.
    a = plot_results._resolve_annotations(events, ["postprocess", "preprocess"])
    b = plot_results._resolve_annotations(events, ["preprocess", "postprocess"])
    assert sorted(a["label"].to_list()) == sorted(b["label"].to_list())
    assert sorted(a["label"].to_list()) == ["postprocess", "preprocess"]

    # Subset that names a scope label picks BOTH its enter and exit events.
    s = plot_results._resolve_annotations(events, ["inference"])
    assert sorted(s["role"].to_list()) == ["enter", "exit"]


def test_missing_directory_errors(tmp_path, capsys):
    plot_results = _load_module()
    rc = plot_results.main([str(tmp_path / "nope")])
    assert rc == 2


def test_empty_samples_returns_zero(tmp_path):
    plot_results = _load_module()
    out = tmp_path / "out"
    out.mkdir()
    pl.DataFrame(
        schema={
            "ts_ns": pl.Int64,
            "source": pl.Utf8,
            "domain": pl.Utf8,
            "value": pl.Float64,
            "kind": pl.Utf8,
        }
    ).write_parquet(out / "samples.parquet")
    rc = plot_results.main([str(out)])
    assert rc == 0


def test_domain_subset(tmp_path):
    plot_results = _load_module()
    out = _write_parquet(tmp_path)
    rc = plot_results.main([str(out), "--domains", "gpu0_util_pct"])
    assert rc == 0
    plots = {p.name for p in (out / "plots").glob("*.png")}
    assert plots == {"gpu0_util_pct.png"}


def test_safe_filename_sanitizes():
    plot_results = _load_module()
    assert plot_results._safe_filename("a/b c") == "a_b_c"
    assert plot_results._safe_filename("cpu0_energy_mj") == "cpu0_energy_mj"
