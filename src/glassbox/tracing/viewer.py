"""Render a saved JSONL trace as a nested rich tree."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.tree import Tree

from glassbox.tracing.schema import Span, Trace
from glassbox.tracing.tracer import load_trace

_TYPE_STYLE = {
    "iteration": "bold cyan",
    "llm": "bold magenta",
    "tool": "bold green",
}


def build_span_tree(spans: list[Span]) -> list[tuple[Span, list[Span]]]:
    """Rebuild parent→children links; return roots as (span, descendants) trees.

    Children are ordered by start_time so the tree matches wall-clock order.
    Orphans (parent_id missing from the file) are treated as roots.
    """
    by_id = {span.span_id: span for span in spans}
    children: dict[str | None, list[Span]] = defaultdict(list)

    for span in spans:
        parent = span.parent_id if span.parent_id in by_id else None
        children[parent].append(span)

    for group in children.values():
        group.sort(key=lambda s: s.start_time)

    def attach(span: Span) -> tuple[Span, list[Any]]:
        kids = [attach(child) for child in children.get(span.span_id, [])]
        return (span, kids)

    return [attach(root) for root in children[None]]


def _brief(value: Any, *, limit: int = 80) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value.replace("\n", "\\n")
    else:
        text = json.dumps(value, default=str)
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def _span_label(span: Span) -> str:
    style = _TYPE_STYLE.get(span.span_type, "bold")
    latency = f"{span.latency_ms:.0f}ms"
    label = f"[{style}]{span.span_type}[/{style}] {span.name}  [dim]{latency}[/dim]"

    if span.span_type == "llm":
        bits: list[str] = []
        if span.model:
            bits.append(span.model)
        if span.prompt_tokens is not None and span.completion_tokens is not None:
            bits.append(f"tok {span.prompt_tokens}→{span.completion_tokens}")
        if bits:
            label += f"  [dim]{' · '.join(bits)}[/dim]"
        if span.outputs.get("tool_calls"):
            names = [
                tc.get("name", "?")
                for tc in span.outputs["tool_calls"]
                if isinstance(tc, dict)
            ]
            label += f"  [yellow]tools={names}[/yellow]"
        elif span.outputs.get("content"):
            label += f'  [dim]"{_brief(span.outputs["content"], limit=60)}"[/dim]'

    elif span.span_type == "tool":
        if span.outputs.get("ok") is False or span.error:
            err = span.error or span.outputs.get("error") or "failed"
            label += f"  [red]ERROR {_brief(err, limit=60)}[/red]"
        elif "output" in span.outputs:
            label += f"  [dim]→ {_brief(span.outputs['output'], limit=60)}[/dim]"

    elif span.span_type == "iteration":
        reason = span.outputs.get("stopped_reason")
        if reason:
            label += f"  [dim]{reason}[/dim]"

    if span.error and span.span_type != "tool":
        label += f"  [red]ERROR {_brief(span.error, limit=60)}[/red]"

    return label


def _add_nodes(tree: Tree, nodes: list[tuple[Span, list[Any]]]) -> None:
    for span, kids in nodes:
        branch = tree.add(_span_label(span))
        if kids:
            _add_nodes(branch, kids)


def render_trace(trace: Trace, *, console: Console | None = None) -> None:
    """Print a nested tree for ``trace`` to the console."""
    console = console or Console()
    roots = build_span_tree(trace.spans)
    header = (
        f"trace [bold]{trace.trace_id}[/bold]  "
        f"[dim]{len(trace.spans)} spans · {trace.created_at.isoformat()}[/dim]"
    )
    tree = Tree(header)
    if not roots:
        tree.add("[dim](no spans)[/dim]")
    else:
        _add_nodes(tree, roots)
    console.print(tree)


def render_trace_file(path: str | Path, *, console: Console | None = None) -> Trace:
    """Load a JSONL trace file and render it."""
    trace = load_trace(path)
    render_trace(trace, console=console)
    return trace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="glassbox view",
        description="Render a glassbox JSONL trace as a nested tree.",
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to a traces/{trace_id}.jsonl file",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = Console()
    if not args.path.is_file():
        console.print(f"[red]Trace not found:[/red] {args.path}")
        return 2
    try:
        render_trace_file(args.path, console=console)
    except Exception as exc:
        console.print(f"[red]Failed to render:[/red] {type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
