"""OpenTelemetry fanout.

Points become `Span.add_event(label, attributes=metadata)` on the currently-
active span. Scopes start a child span via the global tracer; that child is
ended in `on_exit`. If no active span exists, points are silently dropped.

We stash entered spans in a thread-local stack to match enter/exit pairing
across overlapping scopes within the same thread.
"""

from __future__ import annotations

import threading


class OtelFanout:
    name = "otel"

    def __init__(self) -> None:
        from opentelemetry import trace  # pyright: ignore[reportMissingImports]

        self._trace = trace
        self._tracer = trace.get_tracer("pyscope")
        self._tls = threading.local()

    @classmethod
    def is_available(cls) -> bool:
        try:
            import opentelemetry.trace  # pyright: ignore[reportMissingImports]  # noqa: F401
        except Exception:
            return False
        return True

    def _stack(self) -> list:
        s = getattr(self._tls, "stack", None)
        if s is None:
            s = []
            self._tls.stack = s
        return s

    def on_point(self, label: str, metadata: dict) -> None:
        span = self._trace.get_current_span()
        # No-op when there is no active span (recording context).
        if span is None or not span.is_recording():
            return
        span.add_event(label, attributes=metadata or {})

    def on_enter(self, label: str, metadata: dict) -> None:
        cm = self._tracer.start_as_current_span(label, attributes=metadata or {})
        span = cm.__enter__()
        self._stack().append((cm, span))

    def on_exit(self, label: str, metadata: dict) -> None:
        stk = self._stack()
        if not stk:
            return
        cm, _span = stk.pop()
        try:
            cm.__exit__(None, None, None)
        except Exception:
            pass
