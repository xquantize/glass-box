# glassbox

A from-scratch, fully-local LLM agent focused on **traceability** — every model
call, tool call, and loop iteration is recorded as a span you can inspect.

## Stack

- [Ollama](https://ollama.com) (local models; e.g. `qwen2.5`)
- `openai` SDK → `http://localhost:11434/v1` (not the OpenAI cloud API)
- pydantic, rich, pytest

No agent frameworks. No paid APIs.

## Setup

```bash
pip install -e ".[dev]"
```

Requires a running Ollama instance with your model pulled.
