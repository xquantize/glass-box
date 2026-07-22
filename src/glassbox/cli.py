"""CLI entrypoint: run one agent turn against local Ollama and print the result."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from glassbox.agent import DEFAULT_MAX_ITERATIONS, Agent
from glassbox.llm import DEFAULT_MODEL, DEFAULT_TEMPERATURE
from glassbox.tools import ToolRegistry, build_calculator, build_file_tools, build_sql
from glassbox.tools.files import DEFAULT_WORKSPACE
from glassbox.tools.sql import DEFAULT_DB_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="glassbox",
        description="Run the glassbox ReAct agent against local Ollama.",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="User message for the agent. If omitted, read from stdin.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional sampling seed for replayability",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help=f"ReAct loop cap (default: {DEFAULT_MAX_ITERATIONS})",
    )
    parser.add_argument(
        "--traces-dir",
        type=Path,
        default=Path("traces"),
        help="Directory for JSONL traces (default: traces/)",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=DEFAULT_WORKSPACE,
        help=f"Sandbox root for file tools (default: {DEFAULT_WORKSPACE})",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"SQLite database for run_sql (default: {DEFAULT_DB_PATH})",
    )
    return parser


def default_registry(
    *,
    workspace: Path = DEFAULT_WORKSPACE,
    db_path: Path = DEFAULT_DB_PATH,
) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(build_calculator())
    for tool in build_file_tools(workspace):
        registry.register(tool)
    registry.register(build_sql(db_path))
    return registry


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "view":
        from glassbox.tracing.viewer import main as view_main

        return view_main(argv[1:])
    if argv and argv[0] == "eval":
        # evals/ lives at the repo root (not under src/).
        repo_root = Path(__file__).resolve().parents[2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from evals.run_evals import main as eval_main

        return eval_main(argv[1:])

    args = build_parser().parse_args(argv)
    console = Console(stderr=True)

    prompt = args.prompt
    if prompt is None:
        prompt = sys.stdin.read().strip()
    if not prompt:
        console.print("[red]No prompt provided.[/red]")
        console.print(
            "[dim]Usage: glassbox \"prompt\" | glassbox view <trace> | glassbox eval[/dim]"
        )
        return 2

    agent = Agent(
        default_registry(workspace=args.workspace, db_path=args.db),
        model=args.model,
        temperature=args.temperature,
        seed=args.seed,
        max_iterations=args.max_iterations,
        traces_dir=args.traces_dir,
    )

    console.print(
        f"[dim]model={args.model}  max_iterations={args.max_iterations}  "
        f"workspace={args.workspace}  db={args.db}[/dim]"
    )
    try:
        result = agent.run(prompt)
    except Exception as exc:
        console.print(f"[red]Agent failed:[/red] {type(exc).__name__}: {exc}")
        return 1

    out = Console()
    out.print(Panel(result.answer or "[dim](empty answer)[/dim]", title="answer"))
    trace_path = args.traces_dir / (result.trace_id + ".jsonl")
    out.print(
        f"[dim]stopped={result.stopped_reason}  "
        f"iterations={result.iterations}  "
        f"trace={trace_path}[/dim]"
    )
    out.print(f"[dim]view: glassbox view {trace_path}[/dim]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
