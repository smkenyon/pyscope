"""JAX analog of torch_bench.py: GPU-only, CPU-only, and heterogeneous scopes.

What you should see in the summary, identical in shape to the torch example:
- `gpu_only_linalg` — high gpu energy/util, low cpu
- `cpu_only_linalg` — high cpu energy, ~0 gpu energy/util
- `heterogeneous_pattern` — alternating cpu_phase / gpu_phase / cpu_phase_again
  segments under one parent scope; util_pct on the GPU rises and falls.

JAX dispatch is async on GPU — we call `jax.block_until_ready` to bring each
phase's wall-time honest. Without that, the scope would close before the GPU
actually finished, leaving the energy attribution skewed.

Run with:
    uv sync --extra cpu-gpu-bench
    uv run pyscope --output ./out examples/jax_bench.py
"""

from __future__ import annotations

import sys
import time

import pyscope


def _require_jax():
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        print(
            "This example needs jax. Install: uv sync --extra cpu-gpu-bench",
            file=sys.stderr,
        )
        sys.exit(1)
    return jax, jnp


def _gpu_device(jax):
    for d in jax.devices():
        if d.platform.lower() in ("gpu", "cuda", "rocm"):
            return d
    return None


def _cpu_device(jax):
    for d in jax.devices("cpu"):
        return d
    return jax.devices()[0]


def gpu_only_linalg(jax, jnp, target_seconds: float = 5.0) -> None:
    gpu = _gpu_device(jax)
    if gpu is None:
        print("gpu_only_linalg: no JAX GPU device; skipping", file=sys.stderr)
        return

    key = jax.random.PRNGKey(0)
    a = jax.device_put(jax.random.normal(key, (2048, 2048), dtype=jnp.float32), gpu)
    b = jax.device_put(jax.random.normal(key, (2048, 2048), dtype=jnp.float32), gpu)

    @jax.jit
    def step(x, y):
        c = jnp.matmul(x, y)
        u, s, v = jnp.linalg.svd(c, full_matrices=False)
        return jnp.matmul(u * s, v)

    # Warm up JIT.
    a = step(a, b).block_until_ready()

    with pyscope.scope("gpu_only_linalg", target_seconds=target_seconds):
        start = time.monotonic()
        while time.monotonic() - start < target_seconds:
            a = step(a, b)
        a.block_until_ready()


def cpu_only_linalg(jax, jnp, target_seconds: float = 5.0) -> None:
    cpu = _cpu_device(jax)
    key = jax.random.PRNGKey(1)
    a = jax.device_put(jax.random.normal(key, (1024, 1024), dtype=jnp.float32), cpu)
    b = jax.device_put(jax.random.normal(key, (1024, 1024), dtype=jnp.float32), cpu)

    @jax.jit
    def step(x, y):
        c = jnp.matmul(x, y)
        u, s, v = jnp.linalg.svd(c, full_matrices=False)
        return jnp.matmul(u * s, v)

    a = step(a, b).block_until_ready()

    with pyscope.scope("cpu_only_linalg", target_seconds=target_seconds):
        start = time.monotonic()
        while time.monotonic() - start < target_seconds:
            a = step(a, b)
        a.block_until_ready()


def heterogeneous_pattern(jax, jnp) -> None:
    cpu = _cpu_device(jax)
    gpu = _gpu_device(jax)

    key = jax.random.PRNGKey(2)
    a_cpu = jax.device_put(jax.random.normal(key, (1024, 1024), dtype=jnp.float32), cpu)
    b_cpu = jax.device_put(jax.random.normal(key, (1024, 1024), dtype=jnp.float32), cpu)
    if gpu is not None:
        a_gpu = jax.device_put(jax.random.normal(key, (2048, 2048), dtype=jnp.float32), gpu)
        b_gpu = jax.device_put(jax.random.normal(key, (2048, 2048), dtype=jnp.float32), gpu)

    @jax.jit
    def matmul(x, y):
        return jnp.matmul(x, y)

    with pyscope.scope("heterogeneous_pattern"):
        for cycle in range(3):
            with pyscope.scope("cpu_phase", cycle=cycle):
                deadline = time.monotonic() + 1.0
                while time.monotonic() < deadline:
                    a_cpu = matmul(a_cpu, b_cpu)
                a_cpu.block_until_ready()
                pyscope.annotate("cpu_phase_done", cycle=cycle)

            if gpu is not None:
                with pyscope.scope("gpu_phase", cycle=cycle):
                    deadline = time.monotonic() + 1.0
                    while time.monotonic() < deadline:
                        a_gpu = matmul(a_gpu, b_gpu)
                    a_gpu.block_until_ready()
                    pyscope.annotate("gpu_phase_done", cycle=cycle)

            with pyscope.scope("cpu_phase_again", cycle=cycle):
                deadline = time.monotonic() + 0.5
                while time.monotonic() < deadline:
                    a_cpu = matmul(a_cpu, b_cpu)
                a_cpu.block_until_ready()


def main() -> None:
    jax, jnp = _require_jax()
    pyscope.annotate(
        "jax_bench_start",
        devices=[d.platform for d in jax.devices()],
    )

    gpu_only_linalg(jax, jnp)
    cpu_only_linalg(jax, jnp)
    heterogeneous_pattern(jax, jnp)

    pyscope.annotate("jax_bench_done")


if __name__ == "__main__":
    main()
