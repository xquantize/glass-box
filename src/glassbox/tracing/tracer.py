"""JSONL tracer: one trace_id per run, one Span per line under traces/.

Parent nesting uses a ContextVar so nested ``with tracer.span(...)`` blocks
pick up the outer span_id automatically. Explicit ``parent_id`` always wins.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from glassbox.tracing.schema import Span, SpanType, Trace

_current_span_id: ContextVar[str | None] = ContextVar(
    "glassbox_current_span_id", default=None
)


@dataclass
class _OpenSpan:
    """Mutable bag filled while a span context is active; frozen to Span on exit."""

    span_id: str
    parent_id: str | None
    trace_id: str
    name: str
    span_type: SpanType
    start_time: datetime
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    model: str | None = None
    temperature: float | None = None
    seed: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None

    def record_llm(
        self,
        *,
        model: str,
        temperature: float,
        seed: int | None,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        """Copy LLMResponse replay fields onto this span (names/types match)."""
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

    def finish(self, end_time: datetime) -> Span:
        return Span(
            span_id=self.span_id,
            parent_id=self.parent_id,
            trace_id=self.trace_id,
            name=self.name,
            span_type=self.span_type,
            start_time=self.start_time,
            end_time=end_time,
            latency_ms=(end_time - self.start_time).total_seconds() * 1000.0,
            inputs=self.inputs,
            outputs=self.outputs,
            model=self.model,
            temperature=self.temperature,
            seed=self.seed,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            error=self.error,
        )


class Tracer:
    """Owns one trace_id and appends finished spans to traces/{trace_id}.jsonl."""

    def __init__(self, traces_dir: str | Path = "traces") -> None:
        self.trace_id = str(uuid4())
        self.created_at = datetime.now(timezone.utc)
        self.traces_dir = Path(traces_dir)
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.traces_dir / f"{self.trace_id}.jsonl"
        self._spans: list[Span] = []

    @property
    def path(self) -> Path:
        return self._path

    def get_trace(self) -> Trace:
        return Trace(
            trace_id=self.trace_id,
            created_at=self.created_at,
            spans=list(self._spans),
        )

    def _write(self, span: Span) -> None:
        self._spans.append(span)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(span.model_dump_json() + "\n")

    @contextmanager
    def span(
        self,
        name: str,
        span_type: SpanType,
        parent_id: str | None = None,
    ) -> Iterator[_OpenSpan]:
        """Time a block, capture inputs/outputs/error, append one JSONL span on exit.

        If parent_id is omitted, the currently active span (if any) becomes the parent.
        On exception: records error, writes the span, then re-raises.
        """
        resolved_parent = parent_id if parent_id is not None else _current_span_id.get()
        span_id = str(uuid4())
        open_span = _OpenSpan(
            span_id=span_id,
            parent_id=resolved_parent,
            trace_id=self.trace_id,
            name=name,
            span_type=span_type,
            start_time=datetime.now(timezone.utc),
        )
        token = _current_span_id.set(span_id)
        try:
            yield open_span
        except Exception as exc:
            open_span.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            _current_span_id.reset(token)
            self._write(open_span.finish(datetime.now(timezone.utc)))


def load_trace(path: str | Path) -> Trace:
    """Rebuild a Trace from a JSONL file (one Span JSON object per line)."""
    path = Path(path)
    spans: list[Span] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            spans.append(Span.model_validate(json.loads(line)))
    if not spans:
        raise ValueError(f"no spans in {path}")
    return Trace(
        trace_id=spans[0].trace_id,
        created_at=spans[0].start_time,
        spans=spans,
    )
