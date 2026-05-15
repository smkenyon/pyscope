"""Deterministic fake backend used by tests and the sampler smoke check.

Hidden from auto-discovery (underscore prefix). Opt in explicitly:
    Monitor(backends=[FakeBackend()])
"""

from __future__ import annotations

from typing import Iterable

from pyscope.backends.base import Sample


class FakeBackend:
    """Emits a monotonic energy counter and a cycling util reading.

    The energy counter advances by ``energy_step_mj`` each ``read()`` so that
    last-first delta aggregation produces a predictable value across any
    interval. Util cycles through a small fixed list so percentile aggregates
    are easy to assert.
    """

    name = "fake"

    def __init__(
        self,
        energy_step_mj: float = 10.0,
        util_cycle: tuple[float, ...] = (10.0, 50.0, 90.0, 50.0),
    ) -> None:
        self.energy_step_mj = float(energy_step_mj)
        self.util_cycle = tuple(util_cycle)
        self._energy_mj: float = 0.0
        self._tick: int = 0

    @classmethod
    def is_available(cls) -> bool:
        return True

    def read(self) -> Iterable[Sample]:
        self._energy_mj += self.energy_step_mj
        util = self.util_cycle[self._tick % len(self.util_cycle)]
        self._tick += 1
        return [
            ("fake_energy_mj", self._energy_mj, "energy_mj"),
            ("fake_util_pct", util, "util_pct"),
        ]

    def close(self) -> None:
        return None
