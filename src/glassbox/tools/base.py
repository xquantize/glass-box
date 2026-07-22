"""Tool protocol, result type, and registry.

Every tool validates arguments with a pydantic model. Malformed or hallucinated
calls become ToolResult(ok=False, ...) instead of crashing the agent loop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

ArgsT = TypeVar("ArgsT", bound=BaseModel)


class Tool(Protocol):
    """Minimal surface a tool must expose to the registry and agent loop."""

    name: str
    description: str
    parameters: type[BaseModel]

    def run(self, args: BaseModel) -> str:
        """Execute with already-validated args; return a string observation."""
        ...


@dataclass(frozen=True)
class ToolResult:
    """Outcome of one tool invocation — success or a soft failure the model can read."""

    ok: bool
    output: str
    error: str | None = None


@dataclass
class FunctionTool(Generic[ArgsT]):
    """Concrete tool: name + description + pydantic args model + handler."""

    name: str
    description: str
    parameters: type[ArgsT]
    handler: Callable[[ArgsT], str]

    def run(self, args: ArgsT) -> str:
        return self.handler(args)

    def openai_schema(self) -> dict[str, Any]:
        """OpenAI / Ollama tools[] entry derived from the pydantic model."""
        parameters = self.parameters.model_json_schema()
        parameters.pop("title", None)
        parameters.pop("$defs", None)
        parameters.pop("definitions", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }


@dataclass
class ToolRegistry:
    """Name → tool map; builds the tools list for chat() and dispatches calls."""

    _tools: dict[str, FunctionTool[Any]] = field(default_factory=dict)

    def register(self, tool: FunctionTool[Any]) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> FunctionTool[Any] | None:
        return self._tools.get(name)

    def openai_tools(self) -> list[dict[str, Any]]:
        return [t.openai_schema() for t in self._tools.values()]

    def call(self, name: str, arguments: str | dict[str, Any]) -> ToolResult:
        """Validate arguments and run the tool; never raise on bad model output."""
        tool = self._tools.get(name)
        if tool is None:
            available = sorted(self._tools)
            return ToolResult(
                ok=False,
                output="",
                error=f"Unknown tool: {name!r}. Available: {available}",
            )

        parsed = _parse_arguments(arguments)
        if isinstance(parsed, ToolResult):
            return parsed

        try:
            args = tool.parameters.model_validate(parsed)
        except ValidationError as exc:
            return ToolResult(ok=False, output="", error=f"Invalid arguments: {exc}")

        try:
            output = tool.run(args)
        except Exception as exc:
            return ToolResult(
                ok=False,
                output="",
                error=f"{type(exc).__name__}: {exc}",
            )

        return ToolResult(ok=True, output=output, error=None)


def _parse_arguments(
    arguments: str | dict[str, Any],
) -> dict[str, Any] | ToolResult:
    """Turn a tool-call payload into a dict, or a soft-failure ToolResult."""
    if isinstance(arguments, dict):
        return arguments

    try:
        raw = json.loads(arguments) if arguments.strip() else {}
    except json.JSONDecodeError as exc:
        return ToolResult(ok=False, output="", error=f"Invalid JSON arguments: {exc}")

    if not isinstance(raw, dict):
        return ToolResult(
            ok=False,
            output="",
            error=f"Tool arguments must be a JSON object, got {type(raw).__name__}",
        )
    return raw
