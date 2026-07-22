"""Compare two traces by summarizing each run and lining up key fields."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

from glassbox.tracing.schema import Trace
from glassbox.tracing.tracer import load_trace


@dataclass(frozen=True)
class TraceSummary:
    trace_id: str
    user_message: str | None
    model: str | None
    temperature: float | None
    seed: int | None
    iterations: int
    tool_path: tuple[str, ...]
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    answer: str | None
    stopped_reason: str | None
    error_spans: int


@dataclass(frozen=True)
class FieldDiff:
    field: str
    left: object
    right: object

    @property
    def same(self) -> bool:
        return self.left == self.right


def _tool_name(span_name: str) -> str:
    return span_name.removeprefix("tool.") if span_name.startswith("tool.") else span_name


def summarize(trace: Trace) -> TraceSummary:
    """Collapse a Trace into the fields that matter for side-by-side comparison."""
    spans = sorted(trace.spans, key=lambda s: s.start_time)
    run = next((s for s in spans if s.span_type == "run" and s.name == "run"), None)

    user_message: str | None = None
    model: str | None = None
    temperature: float | None = None
    seed: int | None = None
    answer: str | None = None
    stopped_reason: str | None = None
    latency_ms = 0.0

    if run is not None:
        raw_msg = run.inputs.get("user_message")
        user_message = raw_msg if isinstance(raw_msg, str) else None
        raw_model = run.inputs.get("model")
        model = raw_model if isinstance(raw_model, str) else None
        raw_temp = run.inputs.get("temperature")
        temperature = float(raw_temp) if isinstance(raw_temp, (int, float)) else None
        raw_seed = run.inputs.get("seed")
        seed = raw_seed if isinstance(raw_seed, int) else None
        raw_answer = run.outputs.get("answer")
        answer = raw_answer if isinstance(raw_answer, str) else None
        raw_stopped = run.outputs.get("stopped_reason")
        stopped_reason = raw_stopped if isinstance(raw_stopped, str) else None
        latency_ms = run.latency_ms
    else:
        latency_ms = sum(s.latency_ms for s in spans if s.parent_id is None)

    iterations = sum(1 for s in spans if s.span_type == "iteration")
    tool_path = tuple(
        _tool_name(s.name) for s in spans if s.span_type == "tool"
    )
    prompt_tokens = sum(s.prompt_tokens or 0 for s in spans if s.span_type == "llm")
    completion_tokens = sum(
        s.completion_tokens or 0 for s in spans if s.span_type == "llm"
    )
    error_spans = sum(1 for s in spans if s.error)

    if model is None:
        for span in spans:
            if span.span_type == "llm" and span.model:
                model = span.model
                temperature = span.temperature
                seed = span.seed
                break

    if answer is None:
        for span in reversed(spans):
            if span.span_type == "iteration":
                raw = span.outputs.get("answer")
                if isinstance(raw, str) and raw.strip():
                    answer = raw
                    break
            if span.span_type == "llm":
                raw = span.outputs.get("content")
                if isinstance(raw, str) and raw.strip():
                    answer = raw
                    break

    if stopped_reason is None:
        for span in reversed(spans):
            if span.span_type == "iteration":
                raw = span.outputs.get("stopped_reason")
                if isinstance(raw, str):
                    stopped_reason = raw
                    break

    return TraceSummary(
        trace_id=trace.trace_id,
        user_message=user_message,
        model=model,
        temperature=temperature,
        seed=seed,
        iterations=iterations,
        tool_path=tool_path,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
        answer=answer,
        stopped_reason=stopped_reason,
        error_spans=error_spans,
    )


def diff_summaries(left: TraceSummary, right: TraceSummary) -> tuple[FieldDiff, ...]:
    """Compare summaries field-by-field (trace_id is identity, not a diff row)."""
    pairs: list[tuple[str, object, object]] = [
        ("user_message", left.user_message, right.user_message),
        ("model", left.model, right.model),
        ("temperature", left.temperature, right.temperature),
        ("seed", left.seed, right.seed),
        ("iterations", left.iterations, right.iterations),
        ("tool_path", left.tool_path, right.tool_path),
        ("prompt_tokens", left.prompt_tokens, right.prompt_tokens),
        ("completion_tokens", left.completion_tokens, right.completion_tokens),
        ("latency_ms", round(left.latency_ms), round(right.latency_ms)),
        ("stopped_reason", left.stopped_reason, right.stopped_reason),
        ("error_spans", left.error_spans, right.error_spans),
        ("answer", left.answer, right.answer),
    ]
    return tuple(FieldDiff(field=name, left=a, right=b) for name, a, b in pairs)


def _fmt(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, tuple):
        return " → ".join(str(v) for v in value) if value else "(none)"
    if isinstance(value, float):
        return f"{value:g}"
    text = str(value).replace("\n", "\\n")
    if len(text) > 72:
        return text[:71] + "…"
    return text


def render_diff(
    left: TraceSummary,
    right: TraceSummary,
    rows: tuple[FieldDiff, ...],
    *,
    console: Console | None = None,
) -> None:
    console = console or Console()
    table = Table(title="trace diff")
    table.add_column("field", style="bold")
    table.add_column(left.trace_id[:8])
    table.add_column(right.trace_id[:8])
    table.add_column("")

    for row in rows:
        mark = "[green]same[/green]" if row.same else "[red]diff[/red]"
        table.add_row(row.field, _fmt(row.left), _fmt(row.right), mark)

    console.print(table)
    changed = sum(1 for row in rows if not row.same)
    if changed == 0:
        console.print("[green]identical on compared fields[/green]")
    else:
        console.print(f"[yellow]{changed} field(s) differ[/yellow]")


def diff_files(left_path: Path, right_path: Path, *, console: Console | None = None) -> int:
    """Load two JSONL traces, print a diff table. Returns 0 if same, 1 if different."""
    left = summarize(load_trace(left_path))
    right = summarize(load_trace(right_path))
    rows = diff_summaries(left, right)
    render_diff(left, right, rows, console=console)
    return 0 if all(row.same for row in rows) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="glassbox diff",
        description="Compare two glassbox JSONL traces.",
    )
    parser.add_argument("left", type=Path, help="First traces/{id}.jsonl")
    parser.add_argument("right", type=Path, help="Second traces/{id}.jsonl")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = Console()
    for path in (args.left, args.right):
        if not path.is_file():
            console.print(f"[red]Trace not found:[/red] {path}")
            return 2
    try:
        return diff_files(args.left, args.right, console=console)
    except Exception as exc:
        console.print(f"[red]Diff failed:[/red] {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
