"""pyscope CLI: wrap a Python script in a Monitor and print a summary.

Usage:
    uv run pyscope SCRIPT [SCRIPT_ARGS...]

The user's script runs under `runpy.run_path(..., run_name="__main__")`
with `sys.argv` patched. The Monitor is started before the script runs and
stopped in a `finally` block so we get a summary even if the script raises.
The summary is written to stderr so the script's stdout stays clean.
"""

from __future__ import annotations

import logging
import runpy
import sys
from pathlib import Path
from typing import Optional

import typer

import pyscope
from pyscope.monitor import Monitor

app = typer.Typer(
    add_completion=False,
    help="pyscope — energy and resource monitoring for Python.",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)


def _parse_backends(spec: str | None) -> Optional[list]:
    if not spec:
        return None
    from pyscope.backends._fake import FakeBackend

    name_map = {"fake": FakeBackend}
    for name, modpath, clsname in [
        ("psutil_sys", "pyscope.backends.psutil_sys", "PsutilSysBackend"),
        ("zeus_cpu", "pyscope.backends.zeus_cpu", "ZeusCpuBackend"),
        ("zeus_gpu", "pyscope.backends.zeus_gpu", "ZeusGpuBackend"),
        ("nvml_util", "pyscope.backends.nvml_util", "NvmlUtilBackend"),
        ("zeus_soc", "pyscope.backends.zeus_soc", "ZeusSocBackend"),
        ("tdp_fallback", "pyscope.backends.tdp_fallback", "TdpFallbackBackend"),
    ]:
        try:
            mod = __import__(modpath, fromlist=[clsname])
            name_map[name] = getattr(mod, clsname)
        except Exception:
            pass

    names = [n.strip() for n in spec.split(",") if n.strip()]
    unknown = [n for n in names if n not in name_map]
    if unknown:
        raise typer.BadParameter(f"unknown backend(s): {unknown}; known={list(name_map)}")
    instances = []
    for n in names:
        cls = name_map[n]
        if not cls.is_available():
            print(f"pyscope: backend {n!r} unavailable; skipping", file=sys.stderr)
            continue
        instances.append(cls())
    return instances


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    script: Optional[Path] = typer.Argument(
        None, exists=True, dir_okay=False, readable=True, help="Path to the script to run."
    ),
    interval_ms: int = typer.Option(50, "--interval-ms", help="Sampler period in milliseconds"),
    backends: Optional[str] = typer.Option(
        None, "--backends", help="Comma-separated backend names; overrides auto-detect"
    ),
    no_fanout: bool = typer.Option(False, "--no-fanout", help="Disable fanout emitters"),
    output: Optional[Path] = typer.Option(
        None, "--output", help="Write samples.parquet, events.parquet, segments.parquet here"
    ),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress the stdout summary"),
    format: str = typer.Option("table", "--format", help="Summary format: table or json"),
    log_level: str = typer.Option("WARNING", "--log-level", help="Python logging level"),
) -> None:
    """Run SCRIPT under pyscope and print an energy/resource summary."""
    if script is None:
        typer.echo(ctx.get_help())
        raise typer.Exit(code=0)

    logging.basicConfig(level=log_level.upper(), format="%(levelname)s %(name)s: %(message)s")

    backend_instances = _parse_backends(backends)
    monitor = Monitor(
        interval_ms=interval_ms,
        backends=backend_instances,
        fanout=[] if no_fanout else None,
        output_dir=output,
    )

    # Wire as module singleton so the user's `import pyscope` and any calls to
    # `pyscope.annotate(...)` / `pyscope.scope(...)` resolve to this Monitor.
    pyscope.monitor._singleton = monitor  # type: ignore[attr-defined]
    monitor.start()

    saved_argv = sys.argv
    script_args = list(ctx.args or [])
    sys.argv = [str(script)] + script_args
    exc: BaseException | None = None
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as e:
        if e.code not in (0, None):
            exc = e
    except BaseException as e:
        exc = e
    finally:
        sys.argv = saved_argv
        monitor.stop()
        result = monitor.analyze()
        if output is not None:
            result.write_parquet(output)
        if not quiet:
            text = result.summary(format=format)
            print(text, file=sys.stderr)
        if exc is not None:
            raise exc
