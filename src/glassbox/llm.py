"""Thin OpenAI-SDK wrapper pointed at local Ollama.

All LLM traffic stays on the machine: base_url is localhost:11434.
No tracing here — the agent loop records spans around these calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_API_KEY = "ollama"
DEFAULT_MODEL = "qwen2.5:7b"
DEFAULT_TEMPERATURE = 0.0


@dataclass(frozen=True)
class LLMResponse:
    """Raw chat completion plus the knobs and token counts needed to replay a run."""

    response: ChatCompletion
    model: str
    temperature: float
    seed: int | None
    prompt_tokens: int
    completion_tokens: int

    @property
    def content(self) -> str | None:
        """Assistant message text, or None when the model only emitted tool calls."""
        return self.response.choices[0].message.content

    @property
    def tool_calls(self) -> list[ChatCompletionMessageToolCall] | None:
        """Tool calls from the assistant message, or None if none were made."""
        return self.response.choices[0].message.tool_calls


def make_client(
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = DEFAULT_API_KEY,
) -> OpenAI:
    """Build an OpenAI client aimed at a local Ollama OpenAI-compatible server."""
    return OpenAI(base_url=base_url, api_key=api_key)


def _token_counts(completion: ChatCompletion) -> tuple[int, int]:
    """Return (prompt_tokens, completion_tokens), defaulting missing usage to 0."""
    usage = completion.usage
    if usage is None:
        return 0, 0
    prompt = 0 if usage.prompt_tokens is None else int(usage.prompt_tokens)
    completion_tokens = (
        0 if usage.completion_tokens is None else int(usage.completion_tokens)
    )
    return prompt, completion_tokens


def chat(
    messages: list[ChatCompletionMessageParam],
    *,
    tools: list[dict[str, Any]] | None = None,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    seed: int | None = None,
    client: OpenAI | None = None,
) -> LLMResponse:
    """Send a chat completion request; return the raw response + replay metadata.

    `tools` is the OpenAI tools schema list. Pass None when tool calling is unused.
    `seed` is omitted from the request when None (Ollama treats that as unseeded).
    """
    client = client or make_client()

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if seed is not None:
        kwargs["seed"] = seed
    if tools is not None:
        kwargs["tools"] = tools

    completion = client.chat.completions.create(**kwargs)
    prompt_tokens, completion_tokens = _token_counts(completion)

    return LLMResponse(
        response=completion,
        model=model,
        temperature=temperature,
        seed=seed,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
