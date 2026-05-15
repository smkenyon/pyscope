"""NVTX fanout: makes pyscope labels appear in `nsys profile` traces.

Points become `nvtx.mark`; scopes use `nvtx.range_push` / `nvtx.range_pop`.
Pairing is guaranteed by the Monitor's `scope()` context manager — it
always calls `on_exit` in a try/finally even on exception.
"""

from __future__ import annotations


class NvtxFanout:
    name = "nvtx"

    def __init__(self) -> None:
        import nvtx  # pyright: ignore[reportMissingImports]

        self._nvtx = nvtx

    @classmethod
    def is_available(cls) -> bool:
        try:
            import nvtx  # pyright: ignore[reportMissingImports]  # noqa: F401
        except Exception:
            return False
        return True

    def on_point(self, label: str, metadata: dict) -> None:
        self._nvtx.mark(message=label)

    def on_enter(self, label: str, metadata: dict) -> None:
        self._nvtx.range_push(message=label)

    def on_exit(self, label: str, metadata: dict) -> None:
        self._nvtx.range_pop()
