"""Eval case definitions — small, readable checks against AgentResult + Trace."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalCase:
    """One prompt plus the expectations we score after the agent finishes."""

    id: str
    prompt: str
    # All of these substrings must appear in the final answer (case-insensitive).
    expect_answer_contains: tuple[str, ...] = ()
    # Tool names that must appear as tool.* spans in the trace (e.g. "calculator").
    expect_tools: tuple[str, ...] = ()
    # If set, require this exact stopped_reason.
    expect_stopped: str | None = "final_answer"
    max_iterations: int = 8
    notes: str = ""


CASES: tuple[EvalCase, ...] = (
    EvalCase(
        id="calc_precedence",
        prompt="What is (2 + 3) * 4? Use the calculator tool.",
        expect_answer_contains=("20",),
        expect_tools=("calculator",),
        notes="Basic tool use + arithmetic precedence.",
    ),
    EvalCase(
        id="file_roundtrip",
        prompt=(
            "Using the file tools, write exactly the text HELLO-GLASSBOX to "
            "notes/eval.txt, then read that file back and tell me what it contains."
        ),
        expect_answer_contains=("HELLO-GLASSBOX",),
        expect_tools=("write_file", "read_file"),
        notes="Sandbox write then read.",
    ),
    EvalCase(
        id="sql_pets",
        prompt=(
            "Using run_sql: create table pets(name TEXT, age INT) if needed, "
            "insert a cat named Momo age 3, then select all rows. "
            "Tell me the name and age you got back."
        ),
        expect_answer_contains=("Momo", "3"),
        expect_tools=("run_sql",),
        notes="Multi-step SQL via the tool.",
    ),
    EvalCase(
        id="no_tool_hello",
        prompt="Reply with exactly: ping",
        expect_answer_contains=("ping",),
        expect_tools=(),
        max_iterations=3,
        notes="Smoke test that the loop can finish without tools.",
    ),
)


def cases_by_id(*ids: str) -> tuple[EvalCase, ...]:
    """Filter CASES by id; empty ids means all cases."""
    if not ids:
        return CASES
    wanted = set(ids)
    selected = tuple(case for case in CASES if case.id in wanted)
    missing = wanted - {case.id for case in selected}
    if missing:
        raise KeyError(f"unknown eval case id(s): {sorted(missing)}")
    return selected
