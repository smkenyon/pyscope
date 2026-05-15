"""End-to-end CLI tests using a fixture script run under typer's CliRunner."""

from __future__ import annotations

import textwrap

import polars as pl
import pytest
from typer.testing import CliRunner

from pyscope.cli import app
from pyscope.monitor import _reset_singleton_for_tests


@pytest.fixture(autouse=True)
def _reset():
    _reset_singleton_for_tests()
    yield
    _reset_singleton_for_tests()


@pytest.fixture
def fixture_script(tmp_path):
    p = tmp_path / "user_script.py"
    p.write_text(textwrap.dedent("""
        import pyscope
        pyscope.annotate("first_marker")
        with pyscope.scope("preprocess"):
            for _ in range(1000):
                _ = sum(range(100))
        with pyscope.scope("inference"):
            for _ in range(2000):
                _ = sum(range(100))
        pyscope.annotate("done")
    """).strip())
    return p


def test_help_works():
    runner = CliRunner()
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0
    assert "pyscope" in r.stdout


def test_runs_script_and_emits_summary(fixture_script):
    runner = CliRunner()
    r = runner.invoke(
        app,
        ["--interval-ms", "10", "--backends", "fake", str(fixture_script)],
    )
    assert r.exit_code == 0, r.stderr
    # Summary is printed to stderr.
    assert "preprocess" in r.stderr
    assert "inference" in r.stderr


def test_writes_parquet_when_output_given(tmp_path, fixture_script):
    out_dir = tmp_path / "out"
    runner = CliRunner()
    r = runner.invoke(
        app,
        [
            "--interval-ms", "10",
            "--backends", "fake",
            "--output", str(out_dir),
            str(fixture_script),
        ],
    )
    assert r.exit_code == 0, r.stderr
    assert (out_dir / "samples.parquet").exists()
    assert (out_dir / "events.parquet").exists()
    assert (out_dir / "segments.parquet").exists()

    samples = pl.read_parquet(out_dir / "samples.parquet")
    events = pl.read_parquet(out_dir / "events.parquet")
    segments = pl.read_parquet(out_dir / "segments.parquet")
    assert samples.height > 0
    assert "preprocess" in segments["label"].to_list()
    assert "inference" in segments["label"].to_list()
    # Annotations recorded as point events.
    point_labels = events.filter(events["role"] == "point")["label"].to_list()
    assert "first_marker" in point_labels
    assert "done" in point_labels


def test_quiet_suppresses_summary(fixture_script):
    runner = CliRunner()
    r = runner.invoke(
        app,
        ["--interval-ms", "10", "--backends", "fake", "--quiet", str(fixture_script)],
    )
    assert r.exit_code == 0
    assert "preprocess" not in r.stderr
    assert "inference" not in r.stderr


def test_script_exception_still_prints_summary(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text(textwrap.dedent("""
        import pyscope
        with pyscope.scope("crashy"):
            raise RuntimeError("expected")
    """).strip())
    runner = CliRunner()
    r = runner.invoke(
        app,
        ["--interval-ms", "20", "--backends", "fake", str(bad)],
    )
    assert r.exit_code != 0
    # Even with the script crashing, the summary should be printed.
    assert "crashy" in r.stderr


def test_unknown_backend_raises():
    runner = CliRunner()
    r = runner.invoke(app, ["--backends", "totally_made_up", "doesnt_matter.py"])
    assert r.exit_code != 0


def test_no_args_shows_help_and_zero():
    runner = CliRunner()
    r = runner.invoke(app, [])
    assert r.exit_code == 0
    assert "Usage" in r.stdout or "pyscope" in r.stdout
