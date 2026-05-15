"""Segment-builder tests: nesting, points, cross-thread, mismatched exits."""

from __future__ import annotations

from pyscope.analysis import build_segments
from pyscope.events import Event


def _e(ts: int, label: str, role: str, tid: int = 1, **md):
    return Event(ts_ns=ts, label=label, role=role, metadata=md, thread_id=tid)


def test_sequential_pair():
    segs = build_segments([_e(100, "a", "enter"), _e(200, "a", "exit")])
    assert segs.height == 1
    row = segs.row(0, named=True)
    assert row["label"] == "a"
    assert row["t_start_ns"] == 100
    assert row["t_end_ns"] == 200
    assert row["duration_ns"] == 100
    assert row["parent_id"] is None
    assert row["depth"] == 0


def test_nested_pair_assigns_parent_id():
    segs = build_segments([
        _e(0, "outer", "enter"),
        _e(10, "inner", "enter"),
        _e(20, "inner", "exit"),
        _e(30, "outer", "exit"),
    ])
    assert segs.height == 2
    outer = segs.filter(segs["label"] == "outer").row(0, named=True)
    inner = segs.filter(segs["label"] == "inner").row(0, named=True)
    assert inner["parent_id"] == outer["segment_id"]
    assert inner["depth"] == 1
    assert outer["depth"] == 0


def test_point_marker_zero_duration():
    segs = build_segments([
        _e(0, "outer", "enter"),
        _e(15, "tick", "point"),
        _e(30, "outer", "exit"),
    ])
    tick = segs.filter(segs["label"] == "tick").row(0, named=True)
    assert tick["t_start_ns"] == tick["t_end_ns"] == 15
    assert tick["duration_ns"] == 0
    outer = segs.filter(segs["label"] == "outer").row(0, named=True)
    assert tick["parent_id"] == outer["segment_id"]
    assert tick["depth"] == 1


def test_top_level_point_has_no_parent():
    segs = build_segments([_e(7, "p", "point")])
    row = segs.row(0, named=True)
    assert row["parent_id"] is None
    assert row["depth"] == 0


def test_two_separate_threads_independent_stacks():
    segs = build_segments([
        _e(0, "a", "enter", tid=1),
        _e(5, "b", "enter", tid=2),
        _e(10, "a", "exit", tid=1),
        _e(15, "b", "exit", tid=2),
    ])
    a = segs.filter(segs["label"] == "a").row(0, named=True)
    b = segs.filter(segs["label"] == "b").row(0, named=True)
    assert a["thread_id"] == 1
    assert b["thread_id"] == 2
    assert a["parent_id"] is None
    assert b["parent_id"] is None


def test_exit_without_enter_is_dropped():
    segs = build_segments([
        _e(10, "ghost", "exit", tid=1),
        _e(20, "real", "enter", tid=1),
        _e(30, "real", "exit", tid=1),
    ])
    assert segs.height == 1
    assert segs.row(0, named=True)["label"] == "real"


def test_label_mismatch_closes_innermost():
    segs = build_segments([
        _e(0, "a", "enter"),
        _e(10, "b", "enter"),
        # mismatched exit "a" — closes "b" by stack order, with a warning.
        _e(20, "a", "exit"),
    ])
    # "b" got closed because it was on top. "a" never got its exit so it
    # remains open and is not emitted.
    labels = sorted(segs["label"].to_list())
    assert labels == ["b"]


def test_metadata_preserved_on_enter():
    segs = build_segments([
        _e(0, "x", "enter", batch=4, lr=0.01),
        _e(10, "x", "exit"),
    ])
    import json
    meta = json.loads(segs.row(0, named=True)["metadata"])
    assert meta == {"batch": 4, "lr": 0.01}
