"""Tests for CounterUnwrapper."""

from __future__ import annotations

import pytest

from pyscope.wrap import CounterUnwrapper


def test_first_read_returns_raw():
    u = CounterUnwrapper(max_range=1000)
    assert u.feed(42) == 42


def test_monotonic_no_wrap():
    u = CounterUnwrapper(max_range=1000)
    out = [u.feed(v) for v in [10, 20, 30, 40, 999]]
    assert out == [10, 20, 30, 40, 999]


def test_single_wrap():
    u = CounterUnwrapper(max_range=1000)
    out = [u.feed(v) for v in [900, 950, 50, 100]]
    # After the drop 950→50, accumulator adds 1000; 50 becomes 1050; 100 becomes 1100.
    assert out == [900, 950, 1050, 1100]


def test_two_wraps():
    u = CounterUnwrapper(max_range=1000)
    out = [u.feed(v) for v in [900, 50, 800, 100]]
    # 900 → 900
    # drop to 50: +1000, returns 1050
    # 800 ≥ 50: returns 1800
    # drop to 100: +1000, returns 2100
    assert out == [900, 1050, 1800, 2100]


def test_zero_delta_is_no_wrap():
    u = CounterUnwrapper(max_range=1000)
    out = [u.feed(v) for v in [500, 500, 500]]
    assert out == [500, 500, 500]


def test_max_range_edge():
    # Counter sitting right at max-1 then wrapping to 0.
    u = CounterUnwrapper(max_range=100)
    out = [u.feed(v) for v in [99, 0, 1]]
    assert out == [99, 100, 101]


def test_max_range_must_be_positive():
    with pytest.raises(ValueError):
        CounterUnwrapper(max_range=0)
    with pytest.raises(ValueError):
        CounterUnwrapper(max_range=-5)
