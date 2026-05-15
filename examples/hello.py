"""Minimal pyscope demo: nested scopes + point annotations.

Uses numpy (which releases the GIL) so the sampler thread can collect data
even on a CPU-bound workload. With pure-Python loops you can starve the
sampler — see the README's GIL caveat for details.

Run with:
    uv run pyscope examples/hello.py
    uv run pyscope --output ./out examples/hello.py
"""

from __future__ import annotations

import time

import numpy as np

import pyscope


def busy_work(size: int, iters: int) -> float:
    """GIL-releasing numpy work — matmul on float32."""
    a = np.random.randn(size, size).astype(np.float32)
    b = np.random.randn(size, size).astype(np.float32)
    acc = 0.0
    for _ in range(iters):
        c = a @ b
        acc += float(c.sum())
    return acc


pyscope.annotate("startup")

with pyscope.scope("warmup"):
    busy_work(256, 5)

with pyscope.scope("training_loop", epochs=3):
    for epoch in range(3):
        with pyscope.scope("epoch", epoch=epoch):
            pyscope.annotate("epoch_begin", epoch=epoch)
            busy_work(512, 20)
            # A small sleep to make scope durations visible at default 50 ms
            # sampling cadence even on fast machines.
            time.sleep(0.2)
            pyscope.annotate("epoch_end", epoch=epoch)

with pyscope.scope("cleanup"):
    busy_work(256, 5)

pyscope.annotate("shutdown")
