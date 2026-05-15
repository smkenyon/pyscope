"""Build polars DataFrames from collected samples + events and summarize them.

The Monitor stops, collects raw rows, and calls `build_analysis()` to produce
an `AnalysisResult`. The result holds three long-format frames:

- `samples` : (ts_ns, source, domain, value, kind)
- `events`  : (ts_ns, label, role, metadata, thread_id)
- `segments`: (segment_id, label, t_start_ns, t_end_ns, duration_ns,
               parent_id, depth, thread_id, metadata)

Segments are constructed by walking events in timestamp order while
maintaining a stack per thread_id. Bare `point` events become zero-duration
segments. Nested scopes get `parent_id` from the stack at enter time.

Aggregations dispatched by `kind`:
- energy_mj         → last - first within the window (cumulative counter)
- power_mw[_est]    → trapezoidal integral → derived energy_mj
- util_pct          → mean, p50, p95, max
- bytes             → mean, max, peak-vs-baseline (max - first)
"""

from __future__ import annotations

import json
import logging
import platform
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from pyscope.events import Event

log = logging.getLogger("pyscope.analysis")


# --- Raw → DataFrames ---------------------------------------------------

SAMPLE_SCHEMA = {
    "ts_ns": pl.Int64,
    "source": pl.Utf8,
    "domain": pl.Utf8,
    "value": pl.Float64,
    "kind": pl.Utf8,
}

EVENT_SCHEMA = {
    "ts_ns": pl.Int64,
    "label": pl.Utf8,
    "role": pl.Utf8,
    "metadata": pl.Utf8,  # JSON-encoded
    "thread_id": pl.Int64,
}

SEGMENT_SCHEMA = {
    "segment_id": pl.Int64,
    "label": pl.Utf8,
    "t_start_ns": pl.Int64,
    "t_end_ns": pl.Int64,
    "duration_ns": pl.Int64,
    "parent_id": pl.Int64,
    "depth": pl.Int32,
    "thread_id": pl.Int64,
    "metadata": pl.Utf8,
}


def samples_to_df(rows: list[tuple[int, str, str, float, str]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=SAMPLE_SCHEMA)
    return pl.DataFrame(rows, schema=SAMPLE_SCHEMA, orient="row")


def events_to_df(events: list[Event]) -> pl.DataFrame:
    if not events:
        return pl.DataFrame(schema=EVENT_SCHEMA)
    rows = [
        (e.ts_ns, e.label, e.role, json.dumps(e.metadata, default=str), e.thread_id)
        for e in events
    ]
    return pl.DataFrame(rows, schema=EVENT_SCHEMA, orient="row")


def build_segments(events: list[Event]) -> pl.DataFrame:
    """Walk events in ts order, build a segment table.

    Cross-thread `exit` with no matching open scope on its own thread is
    logged and dropped (does not crash).
    """
    if not events:
        return pl.DataFrame(schema=SEGMENT_SCHEMA)

    ordered = sorted(events, key=lambda e: e.ts_ns)
    next_id = 0
    rows: list[tuple[int, str, int, int, int, int | None, int, int, str]] = []
    # Per-thread stack: (segment_id, label, t_start, metadata, parent_id, depth)
    stacks: dict[int, list[tuple[int, str, int, dict[str, Any], int | None, int]]] = {}

    for e in ordered:
        stk = stacks.setdefault(e.thread_id, [])
        if e.role == "enter":
            parent_id = stk[-1][0] if stk else None
            depth = len(stk)
            sid = next_id
            next_id += 1
            stk.append((sid, e.label, e.ts_ns, e.metadata, parent_id, depth))
        elif e.role == "exit":
            if not stk:
                log.warning(
                    "exit without matching enter on thread %d: label=%r — dropped",
                    e.thread_id,
                    e.label,
                )
                continue
            sid, label, t_start, meta, parent_id, depth = stk.pop()
            if label != e.label:
                log.warning(
                    "scope label mismatch on thread %d: enter=%r exit=%r — "
                    "closing innermost scope by stack order",
                    e.thread_id,
                    label,
                    e.label,
                )
            duration = e.ts_ns - t_start
            rows.append(
                (sid, label, t_start, e.ts_ns, duration, parent_id, depth, e.thread_id,
                 json.dumps(meta, default=str))
            )
        else:  # point — zero-duration segment, parented under current top.
            parent_id = stk[-1][0] if stk else None
            depth = len(stk)
            sid = next_id
            next_id += 1
            rows.append(
                (sid, e.label, e.ts_ns, e.ts_ns, 0, parent_id, depth, e.thread_id,
                 json.dumps(e.metadata, default=str))
            )

    if not rows:
        return pl.DataFrame(schema=SEGMENT_SCHEMA)
    df = pl.DataFrame(rows, schema=SEGMENT_SCHEMA, orient="row")
    return df.sort("t_start_ns")


# --- Window aggregations -------------------------------------------------

def _samples_in_window(samples: pl.DataFrame, t0: int, t1: int) -> pl.DataFrame:
    if samples.is_empty():
        return samples
    if t1 == t0:  # zero-duration segment (point) — include exact-match samples.
        return samples.filter(pl.col("ts_ns") == t0)
    return samples.filter((pl.col("ts_ns") >= t0) & (pl.col("ts_ns") <= t1))


def _trapezoid_integral_mj(ts_ns: list[int], power_mw: list[float]) -> float:
    """Integrate power_mw over ts_ns to get energy_mj.

    Energy_mJ = sum( 0.5 * (p_i + p_{i+1}) * (Δt_ns / 1e6) ), since
    mW * ms = µJ … actually: mW × s = mJ, and Δt_ns/1e9 = seconds.
    So: mW * (ns / 1e9) = mW·s / 1e0 … careful: mW * s = 1e-3 W·s = mJ.
    Δt in seconds = Δt_ns / 1e9, so mJ = mW × Δt_ns / 1e9.
    """
    if len(ts_ns) < 2:
        return 0.0
    total = 0.0
    for i in range(len(ts_ns) - 1):
        dt_ns = ts_ns[i + 1] - ts_ns[i]
        if dt_ns <= 0:
            continue
        avg = 0.5 * (power_mw[i] + power_mw[i + 1])
        total += avg * dt_ns / 1e9
    return total


def energy_in_window(samples: pl.DataFrame, t0: int, t1: int) -> dict[str, float]:
    """Compute per-domain energy (mJ) between t0 and t1 (inclusive).

    `energy_mj` kind contributes via last-first delta.
    `power_mw` and `power_mw_estimated` contribute via trapezoidal integration.
    """
    out: dict[str, float] = {}
    win = _samples_in_window(samples, t0, t1)
    if win.is_empty():
        return out

    energy_rows = win.filter(pl.col("kind") == "energy_mj")
    for domain, sub in energy_rows.group_by("domain"):
        domain_str = domain[0] if isinstance(domain, tuple) else domain
        if sub.height < 1:
            continue
        first = sub.row(0)
        last = sub.row(sub.height - 1)
        out[str(domain_str)] = float(last[3]) - float(first[3])

    for kind in ("power_mw", "power_mw_estimated"):
        power_rows = win.filter(pl.col("kind") == kind)
        for domain, sub in power_rows.group_by("domain"):
            domain_str = domain[0] if isinstance(domain, tuple) else domain
            ts = sub["ts_ns"].to_list()
            pw = sub["value"].to_list()
            mj = _trapezoid_integral_mj(ts, pw)
            # Derive an energy entry named for the source.
            key = f"{domain_str}::derived_mj"
            out[key] = mj
    return out


def util_in_window(samples: pl.DataFrame, t0: int, t1: int) -> pl.DataFrame:
    """Return per-domain util stats: mean, p50, p95, max."""
    win = _samples_in_window(samples, t0, t1).filter(pl.col("kind") == "util_pct")
    if win.is_empty():
        return pl.DataFrame(
            schema={"domain": pl.Utf8, "mean": pl.Float64, "p50": pl.Float64,
                    "p95": pl.Float64, "max": pl.Float64}
        )
    return win.group_by("domain").agg(
        pl.col("value").mean().alias("mean"),
        pl.col("value").quantile(0.5).alias("p50"),
        pl.col("value").quantile(0.95).alias("p95"),
        pl.col("value").max().alias("max"),
    )


def bytes_in_window(samples: pl.DataFrame, t0: int, t1: int) -> pl.DataFrame:
    """Return per-domain bytes stats: mean, max, peak_delta (max - first)."""
    win = _samples_in_window(samples, t0, t1).filter(pl.col("kind") == "bytes").sort("ts_ns")
    if win.is_empty():
        return pl.DataFrame(
            schema={"domain": pl.Utf8, "mean": pl.Float64, "max": pl.Float64,
                    "peak_delta": pl.Float64}
        )
    return win.group_by("domain").agg(
        pl.col("value").mean().alias("mean"),
        pl.col("value").max().alias("max"),
        (pl.col("value").max() - pl.col("value").first()).alias("peak_delta"),
    )


# --- Per-segment summary -------------------------------------------------

_CPU_ENERGY_RE = re.compile(r"^cpu\d+_(energy_mj|dram_energy_mj)$")
_GPU_ENERGY_RE = re.compile(r"^gpu\d+_energy_mj$")
_GPU_POWER_RE = re.compile(r"^gpu\d+_power_mw$")
_CPU_POWER_EST_RE = re.compile(r"^cpu.*_power_mw_est$")
_GPU_UTIL_RE = re.compile(r"^gpu\d+_util_pct$")
_VRAM_RE = re.compile(r"^gpu\d+_(proc_vram_bytes|mem_used_bytes)$")
_RAM_DOMAIN = "proc_tree_rss_bytes"
_RAM_DOMAIN_FALLBACK = "system_ram_used_bytes"


def _segment_summary_row(
    seg: dict[str, Any], samples: pl.DataFrame
) -> dict[str, Any]:
    t0 = int(seg["t_start_ns"])
    t1 = int(seg["t_end_ns"])
    win = _samples_in_window(samples, t0, t1)
    cpu_mj = 0.0
    gpu_mj = 0.0
    gpu_util_p95: float | None = None
    ram_peak_b: float | None = None
    vram_peak_b: float | None = None
    has_estimated = False

    if not win.is_empty():
        # Energy: cumulative deltas per matching domain.
        em = win.filter(pl.col("kind") == "energy_mj")
        for domain, sub in em.group_by("domain"):
            domain_str = domain[0] if isinstance(domain, tuple) else domain
            if sub.height < 1:
                continue
            first = sub.row(0)[3]
            last = sub.row(sub.height - 1)[3]
            delta = float(last) - float(first)
            ds = str(domain_str)
            if _CPU_ENERGY_RE.match(ds):
                cpu_mj += delta
            elif _GPU_ENERGY_RE.match(ds):
                gpu_mj += delta

        # Power → derived energy (trapezoid).
        pw = win.filter(pl.col("kind").is_in(["power_mw", "power_mw_estimated"]))
        if not pw.is_empty():
            for domain, sub in pw.group_by("domain"):
                domain_str = domain[0] if isinstance(domain, tuple) else domain
                ts = sub["ts_ns"].to_list()
                vals = sub["value"].to_list()
                mj = _trapezoid_integral_mj(ts, vals)
                ds = str(domain_str)
                if _GPU_POWER_RE.match(ds):
                    gpu_mj += mj
                elif _CPU_POWER_EST_RE.match(ds) or ds.startswith("cpu"):
                    cpu_mj += mj
                if (sub["kind"] == "power_mw_estimated").any():
                    has_estimated = True

        # GPU util p95.
        gu = win.filter(pl.col("kind") == "util_pct")
        gu_rows = gu.filter(pl.col("domain").str.contains(r"^gpu\d+_util_pct$"))
        if not gu_rows.is_empty():
            q = gu_rows["value"].quantile(0.95)
            gpu_util_p95 = float(q) if q is not None else None

        # RAM peak: prefer proc_tree_rss_bytes, fall back to system.
        rb = win.filter(pl.col("kind") == "bytes")
        rss = rb.filter(pl.col("domain") == _RAM_DOMAIN)
        if rss.is_empty():
            rss = rb.filter(pl.col("domain") == _RAM_DOMAIN_FALLBACK)
        if not rss.is_empty():
            mx = rss["value"].max()
            ram_peak_b = float(mx) if isinstance(mx, (int, float)) else None

        vram = rb.filter(pl.col("domain").str.contains(r"^gpu\d+_proc_vram_bytes$"))
        if vram.is_empty():
            vram = rb.filter(pl.col("domain").str.contains(r"^gpu\d+_mem_used_bytes$"))
        if not vram.is_empty():
            mx = vram["value"].max()
            vram_peak_b = float(mx) if isinstance(mx, (int, float)) else None

    duration_ms = (t1 - t0) / 1e6
    return {
        "label": seg["label"],
        "depth": int(seg["depth"]),
        "duration_ms": duration_ms,
        "cpu_energy_J": cpu_mj / 1000.0,
        "gpu_energy_J": gpu_mj / 1000.0,
        "ram_peak_MB": (ram_peak_b or 0.0) / 1e6 if ram_peak_b is not None else None,
        "vram_peak_MB": (vram_peak_b or 0.0) / 1e6 if vram_peak_b is not None else None,
        "gpu_util_p95": gpu_util_p95,
        "source_quality": "estimated" if has_estimated else "measured",
    }


def per_segment_summary(samples: pl.DataFrame, segments: pl.DataFrame) -> pl.DataFrame:
    if segments.is_empty():
        return pl.DataFrame(
            schema={
                "label": pl.Utf8, "depth": pl.Int32, "duration_ms": pl.Float64,
                "cpu_energy_J": pl.Float64, "gpu_energy_J": pl.Float64,
                "ram_peak_MB": pl.Float64, "vram_peak_MB": pl.Float64,
                "gpu_util_p95": pl.Float64, "source_quality": pl.Utf8,
            }
        )
    rows = [_segment_summary_row(seg, samples) for seg in segments.iter_rows(named=True)]
    return pl.DataFrame(rows)


# --- Result object -------------------------------------------------------

@dataclass
class AnalysisResult:
    samples: pl.DataFrame
    events: pl.DataFrame
    segments: pl.DataFrame
    wall_clock_anchor_ns: int = 0
    monotonic_anchor_ns: int = 0

    def energy_between(self, t0_ns: int, t1_ns: int) -> dict[str, float]:
        return energy_in_window(self.samples, t0_ns, t1_ns)

    def energy_in_segment(self, label: str) -> dict[str, float]:
        rows = self.segments.filter(pl.col("label") == label)
        if rows.is_empty():
            return {}
        # If multiple, sum across.
        total: dict[str, float] = {}
        for seg in rows.iter_rows(named=True):
            sub = energy_in_window(self.samples, int(seg["t_start_ns"]), int(seg["t_end_ns"]))
            for k, v in sub.items():
                total[k] = total.get(k, 0.0) + v
        return total

    def util_in_segment(self, label: str) -> pl.DataFrame:
        rows = self.segments.filter(pl.col("label") == label)
        if rows.is_empty():
            return pl.DataFrame()
        # First match only — multiple-segment averaging isn't well-defined here.
        seg = rows.row(0, named=True)
        return util_in_window(self.samples, int(seg["t_start_ns"]), int(seg["t_end_ns"]))

    def summary_frame(self) -> pl.DataFrame:
        return per_segment_summary(self.samples, self.segments)

    def summary(self, format: str = "table") -> str:
        df = self.summary_frame()
        if format == "json":
            return df.write_json()
        return _render_table(df)

    def write_parquet(self, output_dir: Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.samples.write_parquet(output_dir / "samples.parquet")
        self.events.write_parquet(output_dir / "events.parquet")
        self.segments.write_parquet(output_dir / "segments.parquet")


def _render_table(df: pl.DataFrame) -> str:
    """Render summary as a table. Uses rich if available; falls back to text."""
    try:
        from rich.console import Console
        from rich.table import Table
    except Exception:  # pragma: no cover
        return df.__repr__()

    table = Table(title="pyscope summary")
    for col in df.columns:
        if col == "label":
            table.add_column(col, justify="left", no_wrap=True, overflow="fold")
        elif col == "source_quality":
            table.add_column(col, justify="left", no_wrap=True)
        else:
            table.add_column(col, justify="right", no_wrap=True)
    for row in df.iter_rows(named=True):
        formatted = []
        for col in df.columns:
            v = row[col]
            if v is None:
                formatted.append("—")
            elif isinstance(v, float):
                formatted.append(f"{v:.3f}")
            else:
                formatted.append(str(v))
        table.add_row(*formatted)

    # macOS resolution caveat row.
    macos_note = ""
    if platform.system() == "Darwin":
        short = df.filter(pl.col("duration_ms") < 10.0)
        if not short.is_empty():
            macos_note = (
                "\nWarning: Apple Silicon energy counters have ~1 ms resolution; "
                f"{short.height} segment(s) shorter than 10 ms may be unreliable.\n"
            )

    # Capture-only: record=True with a non-tty file means print() goes to the
    # file AND records to the buffer. Use an in-memory file to avoid double-
    # printing — the caller writes the returned string wherever they want.
    import io
    console = Console(record=True, file=io.StringIO(), width=200)
    console.print(table)
    return console.export_text() + macos_note


def build_analysis(
    raw_samples: list[tuple[int, str, str, float, str]],
    events: list[Event],
    wall_clock_anchor_ns: int = 0,
    monotonic_anchor_ns: int = 0,
) -> AnalysisResult:
    samples_df = samples_to_df(raw_samples).sort("ts_ns")
    events_df = events_to_df(events).sort("ts_ns")
    segments_df = build_segments(events)
    return AnalysisResult(
        samples=samples_df,
        events=events_df,
        segments=segments_df,
        wall_clock_anchor_ns=wall_clock_anchor_ns,
        monotonic_anchor_ns=monotonic_anchor_ns,
    )
