"""GPU-aware pyscope demo. Requires CUDA + the `cpu-gpu-bench` extra.

Run with:
    uv sync --extra cpu-gpu-bench
    uv run pyscope examples/gpu_demo.py
"""

from __future__ import annotations

import sys
import time

import pyscope


def _require_cuda():
    try:
        import torch
    except ImportError:
        print("This example needs torch. Install with: uv sync --extra cpu-gpu-bench", file=sys.stderr)
        sys.exit(1)
    if not torch.cuda.is_available():
        print("No CUDA device detected; skipping GPU demo.", file=sys.stderr)
        sys.exit(0)
    return torch


def main() -> None:
    torch = _require_cuda()
    dev = torch.device("cuda")
    pyscope.annotate("cuda_ready", device_name=torch.cuda.get_device_name(0))

    with pyscope.scope("warmup"):
        x = torch.randn(2048, 2048, device=dev)
        torch.matmul(x, x).sum().item()

    with pyscope.scope("training_proxy", iters=50):
        a = torch.randn(4096, 4096, device=dev)
        b = torch.randn(4096, 4096, device=dev)
        for i in range(50):
            with pyscope.scope("step", i=i):
                c = torch.matmul(a, b)
                a = c.clone()
        torch.cuda.synchronize()

    with pyscope.scope("cooldown"):
        time.sleep(0.5)


if __name__ == "__main__":
    main()
