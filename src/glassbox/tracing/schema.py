"""Pydantic models for flat, reconstructable traces.

Spans are stored as a flat list linked by parent_id — never nested on disk.
A viewer walks those links to render a tree.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

SpanType = Literal["llm", "tool", "iteration"]


class Span(BaseModel):
    """One timed unit of work inside a trace.

    LLM-only fields mirror ``LLMResponse`` (model, temperature, seed,
    prompt_tokens, completion_tokens) so an llm span can replay a call when
    the caller fills them from the response.
    """

    span_id: str
    parent_id: str | None
    trace_id: str
    name: str
    span_type: SpanType
    start_time: datetime
    end_time: datetime
    latency_ms: float
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    # Set for span_type == "llm"; left None for tool / iteration spans.
    model: str | None = None
    temperature: float | None = None
    seed: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None


class Trace(BaseModel):
    """A single agent run: flat span list + parent_id links for tree rebuild."""

    trace_id: str
    created_at: datetime
    spans: list[Span] = Field(default_factory=list)
