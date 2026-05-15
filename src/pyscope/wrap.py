"""Monotonic-counter unwrapper for hardware energy counters.

RAPL and similar fixed-width counters wrap when they overflow their range.
`CounterUnwrapper` converts a wrapping raw sequence into a monotonically
increasing accumulated value by detecting drops and adding `max_range`.
"""

from __future__ import annotations


class CounterUnwrapper:
    """Unwraps a fixed-range counter into a monotonic accumulator.

    The counter is assumed to count up modulo ``max_range``. On any read where
    the new raw value is less than the previous raw value, we treat that as a
    single wrap and add ``max_range`` to the accumulator.

    A read equal to the previous value is treated as no wrap (zero delta).
    Multi-wrap-per-read is not detected and would silently undercount; with
    typical RAPL wrap-times measured in minutes and sample intervals in tens
    of milliseconds, this is acceptable.
    """

    __slots__ = ("max_range", "_last_raw", "_accum")

    def __init__(self, max_range: int) -> None:
        if max_range <= 0:
            raise ValueError("max_range must be positive")
        self.max_range = int(max_range)
        self._last_raw: int | None = None
        self._accum: int = 0

    def feed(self, raw: int) -> int:
        """Feed a raw counter reading; return monotonic unwrapped value."""
        raw = int(raw)
        if self._last_raw is None:
            self._last_raw = raw
            return raw
        if raw < self._last_raw:
            self._accum += self.max_range
        self._last_raw = raw
        return self._accum + raw
