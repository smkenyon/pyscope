# pyscope

Energy, power, and resource monitoring for Python code, with NVTX-style
annotations that act as oscilloscope-cursor markers on a continuous sample
timeline.

You drop `pyscope.annotate("label")` or `with pyscope.scope("label"):` markers
in your code. pyscope produces polars DataFrames of timestamped samples and
events plus a stdout summary of per-segment statistics. Annotations also fan
out to NVTX (visible in `nsys profile` traces) and OpenTelemetry (events on
the active span) when those libraries are installed.

## Mental model: oscilloscope cursors

Imagine a sampler thread that records energy/power/util/memory continuously,
like a scope tracing a signal. You place cursor pairs (`scope("region")`)
around regions of interest. pyscope answers *what happened between any two
cursors* — joules consumed, peak RAM, p95 GPU util, wall time, etc.

- `annotate("x")` → a zero-duration point cursor.
- `with scope("x"):` → a cursor pair (`x::enter` / `x::exit`).
- `@monitor.scoped("x")` → decorator form of the scope context manager.

## Install

```bash
uv sync                          # core library
uv sync --extra rich             # nicer summary tables
uv sync --extra nvtx             # forward labels to NVTX
uv sync --extra otel             # forward labels to OpenTelemetry
uv sync --extra cpu-gpu-bench    # torch + torchvision + jax for the GPU examples
```

The package isn't on PyPI yet; install from this repo and run via `uv run pyscope`.

## Use as a library

```python
import pyscope

monitor = pyscope.Monitor(interval_ms=50)
monitor.start()

monitor.annotate("preprocess_done")
with monitor.scope("inference", batch_size=32):
    for batch in batches:
        run(batch)

monitor.stop()
result = monitor.analyze()
print(result.summary())                      # table to stdout
result.write_parquet("./out")                # samples/events/segments parquet
result.energy_in_segment("inference")        # {domain: mJ}
```

The module-level singleton makes one-shot scripts trivial:

```python
import pyscope
pyscope.start()                              # auto-detects backends + fanout
pyscope.annotate("x")
with pyscope.scope("y"): ...
pyscope.stop()
print(pyscope.analyze().summary())
```

## Use as a CLI

```bash
uv run pyscope examples/hello.py
uv run pyscope --interval-ms 25 --output ./out examples/hello.py
uv run pyscope --backends fake examples/hello.py    # for tests
```

Options:

| flag | default | description |
|---|---|---|
| `--interval-ms` | 50 | sampler period |
| `--backends` | auto-detect | comma-separated names; overrides auto-detect |
| `--no-fanout` | off | disable all fanout emitters |
| `--output PATH` | none | write `samples/events/segments.parquet` here |
| `--quiet` | off | suppress the stdout summary |
| `--format` | `table` | `table` or `json` |
| `--log-level` | `WARNING` | python logging level |

## Backends

| name | source | what it emits |
|---|---|---|
| `zeus_cpu` | Intel/AMD RAPL via zeus | `cpu{i}_energy_mj`, `cpu{i}_dram_energy_mj` |
| `zeus_gpu` | NVIDIA NVML / AMD smi via zeus | `gpu{i}_energy_mj` (Volta+) or `gpu{i}_power_mw` fallback |
| `zeus_soc` | Apple Silicon / Jetson via zeus | `soc_{rail}_energy_mj` |
| `nvml_util` | direct pynvml | `gpu{i}_util_pct`, `gpu{i}_mem_used_bytes`, per-PID `gpu{i}_proc_vram_bytes` |
| `psutil_sys` | psutil | `system_ram_*`, `proc_tree_rss_bytes`, `cpu*_util_pct` |
| `tdp_fallback` | bundled vendor table × psutil util | `cpu_total_power_mw_est` (only when `zeus_cpu` is unavailable) |

Auto-detection runs `is_available()` on each backend and instantiates the ones
that pass. Backends that can't initialize log one warning and the tool keeps
running.

## Fanout

| name | when active | what it does |
|---|---|---|
| `nvtx` | `nvtx` installed | `nvtx.mark` for points, `range_push/range_pop` for scopes — visible in `nsys profile` |
| `otel` | `opentelemetry-api` installed | `Span.add_event` on the active span; child span for scopes |
| `perf` | `/tmp/pyscope-perf.fifo` exists | tab-separated lines for `perf script` consumption (opt-in) |

## Known gotchas

- **GIL jitter**: the sampler runs in a Python thread. CPU-bound *pure-Python*
  user code may starve the sampler — `examples/hello.py` uses numpy precisely
  for this reason. Real ML workloads (torch, jax, numpy) release the GIL.
- **RAPL permissions**: `/sys/class/powercap/intel-rapl/.../energy_uj` is
  root-only on most modern Linuxes. `zeus_cpu.is_available()` returns False
  and logs a hint; `tdp_fallback` activates in its place.
- **Containers / VMs**: `/sys/class/powercap` is often empty in unprivileged
  containers and on ARM cloud guests (Graviton). `tdp_fallback` covers these
  with a util×TDP estimate; summary marks these segments `source_quality=estimated`.
- **NVML pre-Volta**: `nvmlDeviceGetTotalEnergyConsumption` raises
  `NotSupported`. `zeus_gpu` silently falls back to integrating `power_mw`.
- **Apple Silicon resolution**: ~1 ms quantization. The summary adds a warning
  for any segment shorter than 10 ms on macOS.
- **Wall-clock vs monotonic**: every timestamp is `time.monotonic_ns()`. A
  single `time.time_ns()` anchor is captured at start for display only — long
  runs may drift relative to NTP-adjusted wall time.
- **Cross-thread scopes**: scope `enter` on thread A and `exit` on thread B
  is undefined behavior. The segment builder logs a warning and closes the
  innermost scope on the exit's own thread.

## What the summary shows

| column | meaning |
|---|---|
| `label` | scope or annotation label |
| `depth` | nesting depth (0 = top-level) |
| `duration_ms` | wall time |
| `cpu_energy_J` | sum of CPU energy domains in the window (delta for `energy_mj`, integral for `power_mw[_estimated]`) |
| `gpu_energy_J` | same, for GPU domains |
| `ram_peak_MB` | max process-tree RSS (falls back to system RAM used) |
| `vram_peak_MB` | max per-process VRAM (falls back to device-wide) |
| `gpu_util_p95` | 95th percentile GPU utilization |
| `source_quality` | `measured` if every contributing source is a hardware counter; `estimated` if any are `power_mw_estimated` (i.e. tdp_fallback) |

## Examples

- `examples/hello.py` — pyscope's smallest meaningful demo. CPU + GPU-idle.
- `examples/gpu_demo.py` — torch matmul loop under a single scope.
- `examples/torch_bench.py` — three scopes: GPU-only linalg (~5 s), CPU-only
  linalg (~5 s), heterogeneous CPU→GPU→CPU pattern. Stays under 4 GB VRAM.
- `examples/torchvision_detector.py` — Faster R-CNN inference with explicit
  `preprocess` / `inference_loop` / `postprocess` scopes and 100 child
  `infer_one` segments under `inference_loop`.
- `examples/jax_bench.py` — JAX analog of `torch_bench.py`, with
  `block_until_ready` calls so each phase's wall time is honest.

Each example writes parquet when `--output DIR` is given; load with
`pl.read_parquet(...)` for plotting and post-hoc analysis.

## Plotting

`plot_results.py` reads the parquet bundle and writes one PNG per sampled
domain into `DIR/plots/`. Annotations and scope boundaries become dotted
vertical lines with rotated labels.

```bash
uv run python plot_results.py ./out
uv run python plot_results.py ./out --annotations preprocess inference postprocess
uv run python plot_results.py ./out --domains gpu0_util_pct cpu_total_power_mw_est
uv run python plot_results.py ./out --output ./out/plots --show
```

`--annotations` takes labels in any order — scope enter/exit pairs are
matched by label, so passing `inference` overlays both the enter and exit
markers. Axes use the right unit per resource (Energy J, Power W,
Utilization %, Memory MB); energy counters are differenced from t0 so the
curve shows cumulative energy spent during the run.

## Out of scope (v1)

Subprocess sampler, IPMI / Redfish / BMC backends, cross-machine monitoring,
realtime Prometheus / OTel-metrics export, web UI, carbon estimates,
power-limit control, Windows.
