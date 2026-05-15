"""Three torch benchmarks under one pyscope run: GPU-only, CPU-only, heterogeneous.

What you should see in the summary:
- `gpu_only_linalg` has high gpu_energy_J, gpu_util_p95 near 100, and modest
  cpu_energy_J. VRAM should stay well below 4 GB.
- `cpu_only_linalg` has high cpu_energy_J and ~0 gpu_energy_J / gpu_util.
- `heterogeneous_pattern` shows alternating CPU and GPU work — the GPU
  energy and util_pct dip during CPU phases and rise during GPU phases.
  Inspect the per-step segments (`step`, `cpu_phase`, `gpu_phase`) for the
  bouncing pattern; the parquet output captures this for plotting.

Run with:
    uv sync --extra cpu-gpu-bench
    uv run pyscope --output ./out examples/torch_bench.py
"""

from __future__ import annotations

import sys
import time

import pyscope


def _require_torch():
    try:
        import torch
    except ImportError:
        print("This example needs torch. Install: uv sync --extra cpu-gpu-bench", file=sys.stderr)
        sys.exit(1)
    return torch


def gpu_only_linalg(torch, target_seconds: float = 5.0) -> None:
    """SVD + matmul loop on GPU only. Targets ~5s wall time. Stays under 4 GB VRAM.

    Two 2048×2048 float32 matrices + intermediates ≈ 2 × 2048² × 4 = ~32 MB
    each; SVD bumps this with workspace but stays well under 4 GB.
    """
    if not torch.cuda.is_available():
        print("gpu_only_linalg: no CUDA device; skipping", file=sys.stderr)
        return
    dev = torch.device("cuda")
    a = torch.randn(2048, 2048, device=dev, dtype=torch.float32)
    b = torch.randn(2048, 2048, device=dev, dtype=torch.float32)
    torch.cuda.synchronize()

    with pyscope.scope("gpu_only_linalg", target_seconds=target_seconds):
        start = time.monotonic()
        i = 0
        while time.monotonic() - start < target_seconds:
            c = torch.matmul(a, b)
            # SVD is heavy; alternate to keep variety in the workload.
            if i % 4 == 0:
                _u, _s, _v = torch.linalg.svd(c, full_matrices=False)
            a = c.contiguous()
            i += 1
        torch.cuda.synchronize()


def cpu_only_linalg(torch, target_seconds: float = 5.0) -> None:
    """Same arithmetic pattern as gpu_only_linalg, but on CPU."""
    dev = torch.device("cpu")
    a = torch.randn(1024, 1024, device=dev, dtype=torch.float32)
    b = torch.randn(1024, 1024, device=dev, dtype=torch.float32)

    with pyscope.scope("cpu_only_linalg", target_seconds=target_seconds):
        start = time.monotonic()
        i = 0
        while time.monotonic() - start < target_seconds:
            c = torch.matmul(a, b)
            if i % 4 == 0:
                _u, _s, _v = torch.linalg.svd(c, full_matrices=False)
            a = c.contiguous()
            i += 1


def heterogeneous_pattern(torch) -> None:
    """One scope, CPU → GPU → CPU phases. Use the segments to see the bounce."""
    has_cuda = torch.cuda.is_available()
    cpu = torch.device("cpu")
    gpu = torch.device("cuda") if has_cuda else None

    a_cpu = torch.randn(1024, 1024, dtype=torch.float32, device=cpu)
    b_cpu = torch.randn(1024, 1024, dtype=torch.float32, device=cpu)
    if has_cuda:
        a_gpu = torch.randn(2048, 2048, dtype=torch.float32, device=gpu)
        b_gpu = torch.randn(2048, 2048, dtype=torch.float32, device=gpu)

    with pyscope.scope("heterogeneous_pattern"):
        for cycle in range(3):
            with pyscope.scope("cpu_phase", cycle=cycle):
                # ~1 s of CPU matmul
                deadline = time.monotonic() + 1.0
                while time.monotonic() < deadline:
                    a_cpu = torch.matmul(a_cpu, b_cpu)
                pyscope.annotate("cpu_phase_done", cycle=cycle)

            if has_cuda:
                with pyscope.scope("gpu_phase", cycle=cycle):
                    # ~1 s of GPU matmul
                    torch.cuda.synchronize()
                    deadline = time.monotonic() + 1.0
                    while time.monotonic() < deadline:
                        a_gpu = torch.matmul(a_gpu, b_gpu)
                    torch.cuda.synchronize()
                    pyscope.annotate("gpu_phase_done", cycle=cycle)

            with pyscope.scope("cpu_phase_again", cycle=cycle):
                deadline = time.monotonic() + 0.5
                while time.monotonic() < deadline:
                    a_cpu = torch.matmul(a_cpu, b_cpu)


def main() -> None:
    torch = _require_torch()
    pyscope.annotate("torch_bench_start", cuda=torch.cuda.is_available())

    gpu_only_linalg(torch)
    cpu_only_linalg(torch)
    heterogeneous_pattern(torch)

    pyscope.annotate("torch_bench_done")


if __name__ == "__main__":
    main()
