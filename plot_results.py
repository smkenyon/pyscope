"""Plot pyscope output: one timeseries per resource, with annotation overlays.

Reads the parquet bundle written by `pyscope --output DIR` (samples.parquet
and events.parquet) and writes one PNG per sampled domain into
`DIR/plots/`.

Usage:
    python plot_results.py [INPUT_DIR]
    python plot_results.py ./out --annotations preprocess inference postprocess
    python plot_results.py ./out --output ./out/plots --show

Notes:
- One resource per plot (no overlays of mixed units).
- Vertical dotted lines mark annotations / scope boundaries; the label text
  rises vertically beside each line.
- `--annotations` is a *set* filter — labels in any order, scope enter/exit
  pairs are matched together by label.
- Axes carry the unit appropriate to the sample `kind` (Energy in J, Power
  in W, Util in %, Memory in MB). Energy counters are differenced from t0
  so the first sample reads as 0 J and the line shows cumulative energy.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import polars as pl

# Use a non-interactive backend by default so headless runs don't block.
matplotlib.use("Agg")


# (ylabel, transform value → display units, title-suffix)
_KIND_AXES = {
    "energy_mj": ("Energy (J, cumulative)", lambda v: v / 1000.0, "energy"),
    "power_mw": ("Power (W)", lambda v: v / 1000.0, "power"),
    "power_mw_estimated": ("Power, estimated (W)", lambda v: v / 1000.0, "power (estimated)"),
    "util_pct": ("Utilization (%)", lambda v: v, "utilization"),
    "bytes": ("Memory (MB)", lambda v: v / 1e6, "memory"),
    "count": ("Count", lambda v: v, "count"),
}


_SAFE_FNAME = re.compile(r"[^A-Za-z0-9_\-]+")


def _safe_filename(domain: str) -> str:
    return _SAFE_FNAME.sub("_", domain)


def _resolve_annotations(events: pl.DataFrame, subset: list[str] | None) -> pl.DataFrame:
    """Pick which events to overlay. If subset is None, plot every point and
    every scope enter/exit. Otherwise filter by label set membership."""
    if events.is_empty():
        return events
    if subset is None:
        return events
    wanted = set(subset)
    return events.filter(pl.col("label").is_in(list(wanted)))


def _format_annotation_text(label: str, role: str) -> str:
    if role == "enter":
        return f"{label} ↳"
    if role == "exit":
        return f"↲ {label}"
    return label


def _annotation_color(role: str) -> str:
    return {"enter": "tab:green", "exit": "tab:red", "point": "tab:purple"}.get(role, "tab:gray")


def _plot_one_domain(
    out_dir: Path,
    domain: str,
    series: pl.DataFrame,
    annotations: pl.DataFrame,
    t0_ns: int,
    show: bool,
) -> Path:
    kind = series["kind"].first()
    assert isinstance(kind, str), f"sample {domain} has non-str kind={kind!r}"
    ylabel, transform, title_kind = _KIND_AXES.get(
        kind, (f"Value ({kind})", lambda v: v, kind)
    )

    t_s = ((series["ts_ns"] - t0_ns) / 1e9).to_list()
    v = [transform(x) for x in series["value"].to_list()]

    # For cumulative energy counters, plot as delta-from-first so the curve
    # represents energy spent during the run, not the device's lifetime total.
    if kind == "energy_mj" and v:
        baseline = v[0]
        v = [x - baseline for x in v]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(t_s, v, linewidth=1.4)
    ax.set_xlabel("Time since start (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{domain} — {title_kind}")
    ax.grid(True, alpha=0.3)

    if t_s:
        ax.set_xlim(t_s[0], t_s[-1] if t_s[-1] > t_s[0] else t_s[0] + 1.0)

    # Annotation markers.
    y_lo, y_hi = ax.get_ylim()
    span = y_hi - y_lo if y_hi > y_lo else 1.0
    text_y = y_lo + span * 0.02

    for evt in annotations.iter_rows(named=True):
        x = (evt["ts_ns"] - t0_ns) / 1e9
        if t_s and (x < t_s[0] or x > t_s[-1]):
            continue
        role = evt["role"]
        ax.axvline(
            x,
            linestyle=":",
            linewidth=1.0,
            color=_annotation_color(role),
            alpha=0.7,
        )
        ax.text(
            x,
            text_y,
            _format_annotation_text(evt["label"], role),
            rotation=90,
            verticalalignment="bottom",
            horizontalalignment="right",
            fontsize=8,
            color=_annotation_color(role),
            alpha=0.85,
        )

    fig.tight_layout()
    out_path = out_dir / f"{_safe_filename(domain)}.png"
    fig.savefig(out_path, dpi=120)
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "input_dir",
        nargs="?",
        default="./out",
        type=Path,
        help="Directory containing samples.parquet and events.parquet (default: ./out)",
    )
    parser.add_argument(
        "--annotations",
        nargs="*",
        default=None,
        help="Subset of annotation labels to overlay. Order doesn't matter; "
        "scope enter/exit pairs are matched by label.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory to write PNGs (default: <input_dir>/plots/)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Pop up plots interactively in addition to writing PNGs.",
    )
    parser.add_argument(
        "--domains",
        nargs="*",
        default=None,
        help="Subset of domains to plot (default: all).",
    )
    args = parser.parse_args(argv)

    input_dir: Path = args.input_dir
    if not input_dir.is_dir():
        print(f"plot_results: {input_dir} is not a directory", file=sys.stderr)
        return 2
    samples_path = input_dir / "samples.parquet"
    events_path = input_dir / "events.parquet"
    if not samples_path.exists():
        print(f"plot_results: missing {samples_path}", file=sys.stderr)
        return 2

    samples = pl.read_parquet(samples_path).sort("ts_ns")
    events = (
        pl.read_parquet(events_path).sort("ts_ns")
        if events_path.exists()
        else pl.DataFrame()
    )

    if samples.is_empty():
        print("plot_results: samples.parquet is empty; nothing to plot", file=sys.stderr)
        return 0

    overlay = _resolve_annotations(events, args.annotations)

    out_dir = args.output or (input_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.show:
        matplotlib.use("TkAgg", force=True)

    t0_ns = int(samples["ts_ns"].min())  # type: ignore[arg-type]

    domains: list[str] = (
        samples["domain"].unique().sort().to_list()
        if args.domains is None
        else [d for d in args.domains if d in samples["domain"].unique().to_list()]
    )
    if not domains:
        print("plot_results: no domains matched", file=sys.stderr)
        return 1

    written = []
    for domain in domains:
        series = samples.filter(pl.col("domain") == domain)
        if series.is_empty():
            continue
        path = _plot_one_domain(
            out_dir=out_dir,
            domain=domain,
            series=series,
            annotations=overlay,
            t0_ns=t0_ns,
            show=args.show,
        )
        written.append(path)

    print(f"plot_results: wrote {len(written)} plot(s) to {out_dir}")
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
