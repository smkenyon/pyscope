"""Tests for the event log and per-thread marker helpers."""

from __future__ import annotations

import threading

from pyscope.events import Event, EventLog


def test_point_records_role_and_thread():
    log = EventLog()
    log.point("hello", {"k": 1})
    snap = log.snapshot()
    assert len(snap) == 1
    e = snap[0]
    assert isinstance(e, Event)
    assert e.label == "hello"
    assert e.role == "point"
    assert e.metadata == {"k": 1}
    assert e.thread_id == threading.get_ident()
    assert e.ts_ns > 0


def test_enter_exit_pair():
    log = EventLog()
    log.enter("blk", {"batch": 32})
    log.exit("blk")
    snap = log.snapshot()
    assert [e.role for e in snap] == ["enter", "exit"]
    assert snap[0].label == snap[1].label == "blk"
    assert snap[1].ts_ns >= snap[0].ts_ns
    # Metadata copied so caller mutations don't leak in.
    assert snap[0].metadata == {"batch": 32}


def test_metadata_is_copied_not_aliased():
    log = EventLog()
    md = {"x": 1}
    log.point("p", md)
    md["x"] = 999
    assert log.snapshot()[0].metadata == {"x": 1}


def test_append_from_multiple_threads_preserves_count():
    log = EventLog()
    N_THREADS = 8
    PER_THREAD = 200
    start = threading.Barrier(N_THREADS)

    def worker(i: int) -> None:
        start.wait()
        for j in range(PER_THREAD):
            log.point(f"t{i}-{j}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    snap = log.snapshot()
    assert len(snap) == N_THREADS * PER_THREAD
    # Every thread's events should be present in monotonic order on its own thread.
    by_thread: dict[int, list[int]] = {}
    for e in snap:
        by_thread.setdefault(e.thread_id, []).append(e.ts_ns)
    assert len(by_thread) == N_THREADS
    for ts_list in by_thread.values():
        assert ts_list == sorted(ts_list)


def test_len_matches_snapshot():
    log = EventLog()
    for i in range(5):
        log.point(f"e{i}")
    assert len(log) == 5
    assert len(log.snapshot()) == 5
