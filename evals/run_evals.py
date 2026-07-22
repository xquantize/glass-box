"""Run glassbox eval cases against local Ollama and print a pass/fail table.

Usage:
  python -m evals.run_evals
  python -m evals.run_evals calc_precedence sql_pets
  glassbox eval
"""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

from evals.cases import EvalCase, cases_by_id
from glassbox.agent import Agent, AgentResult
from glassbox.llm import DEFAULT_MODEL, DEFAULT_TEMPERATURE
from glassbox.tools import ToolRegistry, build_calculator, build_file_tools, build_sql
from glassbox.tracing.schema import Trace
from glassbox.tracing.tracer import Tracer, load_trace


@dataclass(frozen=True)
class CaseResult:
    case: EvalCase
    passed: bool
    failures: tuple[str, ...]
    answer: str
    trace_id: str
    iterations: int
    stopped_reason: str


def build_registry(workspace: Path, db_path: Path) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(build_calculator())
    for tool in build_file_tools(workspace):
        registry.register(tool)
    registry.register(build_sql(db_path))
    return registry


def tools_used(trace: Trace) -> set[str]:
    """Tool span names are ``tool.{name}`` — return the bare tool names."""
    names: set[str] = set()
    for span in trace.spans:
        if span.span_type != "tool":
            continue
        if span.name.startswith("tool."):
            names.add(span.name.removeprefix("tool."))
        else:
            names.add(span.name)
    return names


def score(case: EvalCase, result: AgentResult, trace: Trace) -> tuple[bool, tuple[str, ...]]:
    failures: list[str] = []
    answer_l = result.answer.lower()

    if case.expect_stopped is not None and result.stopped_reason != case.expect_stopped:
        failures.append(
            f"stopped_reason={result.stopped_reason!r}, expected {case.expect_stopped!r}"
        )

    for needle in case.expect_answer_contains:
        if needle.lower() not in answer_l:
            failures.append(f"answer missing {needle!r}")

    used = tools_used(trace)
    for tool in case.expect_tools:
        if tool not in used:
            failures.append(f"tool not used: {tool!r} (saw {sorted(used)})")

    return (not failures, tuple(failures))


def run_case(
    case: EvalCase,
    *,
    model: str,
    temperature: float,
    seed: int | None,
    traces_dir: Path,
) -> CaseResult:
    with tempfile.TemporaryDirectory(prefix=f"glassbox-eval-{case.id}-") as tmp:
        root = Path(tmp)
        workspace = root / "workspace"
        db_path = root / "eval.db"
        workspace.mkdir()

        agent = Agent(
            build_registry(workspace, db_path),
            model=model,
            temperature=temperature,
            seed=seed,
            max_iterations=case.max_iterations,
            traces_dir=traces_dir,
        )
        tracer = Tracer(traces_dir)
        result = agent.run(case.prompt, tracer=tracer)
        trace = load_trace(tracer.path)
        passed, failures = score(case, result, trace)

        return CaseResult(
            case=case,
            passed=passed,
            failures=failures,
            answer=result.answer,
            trace_id=result.trace_id,
            iterations=result.iterations,
            stopped_reason=result.stopped_reason,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="glassbox eval",
        description="Run glassbox eval cases against local Ollama.",
    )
    parser.add_argument(
        "case_ids",
        nargs="*",
        help="Optional case ids to run (default: all).",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--traces-dir",
        type=Path,
        default=Path("traces"),
        help="Where to write per-case JSONL traces.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = Console()

    try:
        cases = cases_by_id(*args.case_ids)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2

    args.traces_dir.mkdir(parents=True, exist_ok=True)
    results: list[CaseResult] = []

    console.print(
        f"[dim]model={args.model}  seed={args.seed}  cases={len(cases)}[/dim]"
    )

    for case in cases:
        console.print(f"\n[bold]→ {case.id}[/bold]  [dim]{case.notes}[/dim]")
        try:
            case_result = run_case(
                case,
                model=args.model,
                temperature=args.temperature,
                seed=args.seed,
                traces_dir=args.traces_dir,
            )
        except Exception as exc:
            case_result = CaseResult(
                case=case,
                passed=False,
                failures=(f"{type(exc).__name__}: {exc}",),
                answer="",
                trace_id="",
                iterations=0,
                stopped_reason="error",
            )
        results.append(case_result)
        status = "[green]PASS[/green]" if case_result.passed else "[red]FAIL[/red]"
        console.print(f"  {status}  iterations={case_result.iterations}")
        if case_result.failures:
            for failure in case_result.failures:
                console.print(f"    [red]- {failure}[/red]")
        if case_result.trace_id:
            console.print(
                f"    [dim]trace=traces/{case_result.trace_id}.jsonl[/dim]"
            )

    table = Table(title="glassbox evals")
    table.add_column("case")
    table.add_column("result")
    table.add_column("iters", justify="right")
    table.add_column("stopped")
    for item in results:
        table.add_row(
            item.case.id,
            "[green]PASS[/green]" if item.passed else "[red]FAIL[/red]",
            str(item.iterations),
            item.stopped_reason,
        )
    console.print()
    console.print(table)

    passed = sum(1 for item in results if item.passed)
    console.print(f"[bold]{passed}/{len(results)} passed[/bold]")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
