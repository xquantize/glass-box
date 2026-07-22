"""Rebuild an Agent run from a saved trace's ``run`` span.

Replay re-executes the same user message with the recorded model / temperature /
seed / max_iterations. It does not cassette LLM bytes — Ollama is called again.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from glassbox.agent import DEFAULT_MAX_ITERATIONS
from glassbox.llm import DEFAULT_MODEL, DEFAULT_TEMPERATURE
from glassbox.tracing.schema import Trace
from glassbox.tracing.tracer import load_trace


@dataclass(frozen=True)
class ReplaySource:
    """Knobs extracted from a trace so a run can be repeated."""

    user_message: str
    model: str
    temperature: float
    seed: int | None
    max_iterations: int
    source_trace_id: str


def extract_replay_source(trace: Trace) -> ReplaySource:
    """Read prompt + sampling knobs from the top-level ``run`` span."""
    run_spans = [s for s in trace.spans if s.span_type == "run" and s.name == "run"]
    if not run_spans:
        raise ValueError(
            "trace has no run span — re-run the agent once so the prompt is recorded, "
            "then replay"
        )
    run = min(run_spans, key=lambda s: s.start_time)
    inputs = run.inputs

    user_message = inputs.get("user_message")
    if not isinstance(user_message, str) or not user_message.strip():
        raise ValueError("run span is missing a user_message")

    model = inputs.get("model")
    if not isinstance(model, str) or not model:
        model = _first_llm_field(trace, "model") or DEFAULT_MODEL

    temperature = inputs.get("temperature")
    if not isinstance(temperature, (int, float)):
        raw = _first_llm_field(trace, "temperature")
        temperature = float(raw) if isinstance(raw, (int, float)) else DEFAULT_TEMPERATURE
    else:
        temperature = float(temperature)

    seed = inputs.get("seed")
    if seed is not None and not isinstance(seed, int):
        seed = None

    max_iterations = inputs.get("max_iterations")
    if not isinstance(max_iterations, int) or max_iterations < 1:
        max_iterations = DEFAULT_MAX_ITERATIONS

    return ReplaySource(
        user_message=user_message,
        model=model,
        temperature=temperature,
        seed=seed,
        max_iterations=max_iterations,
        source_trace_id=trace.trace_id,
    )


def load_replay_source(path: str | Path) -> ReplaySource:
    return extract_replay_source(load_trace(path))


def _first_llm_field(trace: Trace, field: str) -> object | None:
    for span in sorted(trace.spans, key=lambda s: s.start_time):
        if span.span_type != "llm":
            continue
        value = getattr(span, field, None)
        if value is not None:
            return value
    return None
