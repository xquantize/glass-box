"""Safe arithmetic calculator tool (stdlib ast only — no eval of arbitrary code)."""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from glassbox.tools.base import FunctionTool

_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class CalculatorArgs(BaseModel):
    expression: str = Field(
        description=(
            "Arithmetic expression using + - * / // % ** and parentheses, "
            "e.g. '(2 + 3) * 4'."
        )
    )


def _eval_node(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported operator: {type(node.op).__name__}")
        return op(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported unary operator: {type(node.op).__name__}")
        return op(_eval_node(node.operand))
    raise ValueError(f"unsupported expression node: {type(node).__name__}")


def safe_calculate(expression: str) -> str:
    """Parse and evaluate a numeric expression; raise ValueError on anything else."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"could not parse expression: {exc}") from exc
    return str(_eval_node(tree))


def _run(args: CalculatorArgs) -> str:
    return safe_calculate(args.expression)


def build_calculator() -> FunctionTool[CalculatorArgs]:
    """Construct the calculator tool (ready to register)."""
    return FunctionTool(
        name="calculator",
        description="Evaluate a basic arithmetic expression and return the numeric result.",
        parameters=CalculatorArgs,
        handler=_run,
    )
