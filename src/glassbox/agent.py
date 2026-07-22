"""Hand-written ReAct loop — the centerpiece of glassbox.

Each iteration: think (LLM) → optionally act (tools) → observe → repeat.
Every iteration, LLM call, and tool call emits a span. Max-iteration guard
prevents runaway loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)

from glassbox.llm import DEFAULT_MODEL, DEFAULT_TEMPERATURE, LLMResponse, chat, make_client
from glassbox.tools.base import ToolRegistry, ToolResult
from glassbox.tracing.tracer import Tracer

DEFAULT_MAX_ITERATIONS = 8

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful assistant with access to tools. "
    "Use a tool when it helps you answer accurately. "
    "When you have enough information, give a final answer without calling tools."
)

StopReason = Literal["final_answer", "max_iterations"]


@dataclass(frozen=True)
class AgentResult:
    """Outcome of one agent.run() call."""

    answer: str
    trace_id: str
    iterations: int
    stopped_reason: StopReason
    messages: list[ChatCompletionMessageParam]


class Agent:
    """ReAct agent over a ToolRegistry, traced to JSONL."""

    def __init__(
        self,
        tools: ToolRegistry,
        *,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        seed: int | None = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        traces_dir: str | Path = "traces",
        client: OpenAI | None = None,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        self.tools = tools
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt
        self.traces_dir = Path(traces_dir)
        self.client = client or make_client()

    def run(self, user_message: str, *, tracer: Tracer | None = None) -> AgentResult:
        """Run the ReAct loop until a final answer or the iteration cap."""
        tracer = tracer or Tracer(self.traces_dir)
        with tracer.span("run", "run") as run_span:
            run_span.inputs = {
                "user_message": user_message,
                "model": self.model,
                "temperature": self.temperature,
                "seed": self.seed,
                "max_iterations": self.max_iterations,
            }
            result = self._run_loop(user_message, tracer)
            run_span.outputs = {
                "answer": result.answer,
                "stopped_reason": result.stopped_reason,
                "iterations": result.iterations,
            }
            return result

    def _run_loop(self, user_message: str, tracer: Tracer) -> AgentResult:
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]
        tool_schemas = self.tools.openai_tools() or None
        answer = ""
        stopped: StopReason = "max_iterations"
        completed_iterations = 0

        for iteration in range(1, self.max_iterations + 1):
            completed_iterations = iteration
            with tracer.span(f"iteration-{iteration}", "iteration") as iter_span:
                iter_span.inputs = {"iteration": iteration, "message_count": len(messages)}

                response = self._call_llm(tracer, messages, tool_schemas)
                assistant_msg = _assistant_message(response)
                messages.append(assistant_msg)

                tool_calls = response.tool_calls
                if not tool_calls:
                    answer = response.content or ""
                    stopped = "final_answer"
                    iter_span.outputs = {
                        "stopped_reason": stopped,
                        "answer": answer,
                    }
                    break

                observations: list[dict[str, Any]] = []
                for tool_call in tool_calls:
                    result = self._call_tool(tracer, tool_call)
                    tool_message: ChatCompletionMessageParam = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": _tool_observation(result),
                    }
                    messages.append(tool_message)
                    observations.append(
                        {
                            "tool": tool_call.function.name,
                            "ok": result.ok,
                            "output": result.output,
                            "error": result.error,
                        }
                    )

                iter_span.outputs = {
                    "stopped_reason": "continue",
                    "tool_calls": len(tool_calls),
                    "observations": observations,
                }
        else:
            answer = _last_assistant_text(messages) or (
                f"Stopped after {self.max_iterations} iterations without a final answer."
            )
            stopped = "max_iterations"

        return AgentResult(
            answer=answer,
            trace_id=tracer.trace_id,
            iterations=completed_iterations,
            stopped_reason=stopped,
            messages=messages,
        )

    def _call_llm(
        self,
        tracer: Tracer,
        messages: list[ChatCompletionMessageParam],
        tool_schemas: list[dict[str, Any]] | None,
    ) -> LLMResponse:
        with tracer.span("llm.chat", "llm") as llm_span:
            llm_span.inputs = {
                "model": self.model,
                "temperature": self.temperature,
                "seed": self.seed,
                "message_count": len(messages),
                "tools": None if tool_schemas is None else [t["function"]["name"] for t in tool_schemas],
            }
            response = chat(
                messages,
                tools=tool_schemas,
                model=self.model,
                temperature=self.temperature,
                seed=self.seed,
                client=self.client,
            )
            llm_span.record_llm(
                model=response.model,
                temperature=response.temperature,
                seed=response.seed,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
            )
            llm_span.outputs = {
                "content": response.content,
                "tool_calls": _tool_calls_summary(response),
            }
            return response

    def _call_tool(
        self,
        tracer: Tracer,
        tool_call: ChatCompletionMessageToolCall,
    ) -> ToolResult:
        name = tool_call.function.name
        arguments = tool_call.function.arguments
        with tracer.span(f"tool.{name}", "tool") as tool_span:
            tool_span.inputs = {
                "tool_call_id": tool_call.id,
                "name": name,
                "arguments": arguments,
            }
            result = self.tools.call(name, arguments)
            tool_span.outputs = {
                "ok": result.ok,
                "output": result.output,
                "error": result.error,
            }
            if not result.ok and result.error:
                tool_span.error = result.error
            return result


def _assistant_message(response: LLMResponse) -> ChatCompletionMessageParam:
    """Convert an LLMResponse into a chat message dict for the next turn."""
    message = response.response.choices[0].message
    payload: dict[str, Any] = {
        "role": "assistant",
        "content": message.content,
    }
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in message.tool_calls
        ]
    return cast(ChatCompletionMessageParam, payload)


def _tool_observation(result: ToolResult) -> str:
    """String the model sees as the tool result (errors included, soft-fail)."""
    if result.ok:
        return result.output
    return f"ERROR: {result.error}"


def _tool_calls_summary(response: LLMResponse) -> list[dict[str, str]] | None:
    if not response.tool_calls:
        return None
    return [
        {
            "id": tc.id,
            "name": tc.function.name,
            "arguments": tc.function.arguments,
        }
        for tc in response.tool_calls
    ]


def _last_assistant_text(messages: list[ChatCompletionMessageParam]) -> str:
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
    return ""
