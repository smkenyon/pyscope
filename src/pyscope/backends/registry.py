"""Name-keyed backend registry.

Backends are looked up by string name so they can be passed across a
subprocess boundary (Monitor → sampler_main) without pickling instances.
Built-in backends register themselves on import; tests can register
additional names via ``register("flaky", FlakyBackend)``.

Construction goes through ``construct(name, target_pid=...)``; every
registered Backend constructor accepts ``target_pid`` (and may ignore it).
"""

from __future__ import annotations

import logging
from typing import Callable

from pyscope.backends.base import Backend

log = logging.getLogger("pyscope.backends.registry")

_REGISTRY: dict[str, Callable[..., Backend]] = {}


def register(name: str, factory: Callable[..., Backend]) -> None:
    if name in _REGISTRY and _REGISTRY[name] is not factory:
        log.debug("overriding backend %r in registry", name)
    _REGISTRY[name] = factory


def get(name: str) -> Callable[..., Backend]:
    if name not in _REGISTRY:
        raise KeyError(f"unknown backend {name!r}; known={sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available_names() -> list[str]:
    return sorted(_REGISTRY)


def construct(name: str, *, target_pid: int | None = None) -> Backend:
    factory = get(name)
    try:
        return factory(target_pid=target_pid)
    except TypeError:
        return factory()


def _register_builtins() -> None:
    """Import + register every built-in backend. Failures are swallowed so
    a missing optional dep (pynvml, zeus-apple-silicon) doesn't break the
    registry as a whole."""
    for name, modpath, clsname in [
        ("fake", "pyscope.backends._fake", "FakeBackend"),
        ("flaky", "pyscope.backends._fake", "FlakyBackend"),
        ("psutil_sys", "pyscope.backends.psutil_sys", "PsutilSysBackend"),
        ("zeus_cpu", "pyscope.backends.zeus_cpu", "ZeusCpuBackend"),
        ("zeus_gpu", "pyscope.backends.zeus_gpu", "ZeusGpuBackend"),
        ("nvml_util", "pyscope.backends.nvml_util", "NvmlUtilBackend"),
        ("zeus_soc", "pyscope.backends.zeus_soc", "ZeusSocBackend"),
        ("tdp_fallback", "pyscope.backends.tdp_fallback", "TdpFallbackBackend"),
    ]:
        try:
            mod = __import__(modpath, fromlist=[clsname])
            register(name, getattr(mod, clsname))
        except Exception:
            log.debug("failed to register backend %s", name, exc_info=True)


_register_builtins()
