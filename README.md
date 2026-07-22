# glassbox

A from-scratch, fully-local LLM agent focused on **traceability** — every model
call, tool call, and loop iteration is recorded as a span you can inspect.

## Stack

- [Ollama](https://ollama.com) (local models; e.g. `qwen2.5:7b`)
- `openai` SDK → `http://localhost:11434/v1` (not the OpenAI cloud API)
- pydantic, rich, pytest

No agent frameworks. No paid APIs.

## Setup

```bash
pip install -e ".[dev]"
```

Requires a running Ollama instance with your model pulled.

## Run

```bash
glassbox "What is (2 + 3) * 4? Use the calculator."
# or
python -m glassbox "What is (2 + 3) * 4?"
```

File tools are sandboxed to `./workspace` (`--workspace`). SQL uses SQLite at
`./data/glassbox.db` (`--db`). A JSONL trace is written under `traces/` for each run.

```bash
glassbox view traces/<trace_id>.jsonl
glassbox replay traces/<trace_id>.jsonl
glassbox diff traces/<a>.jsonl traces/<b>.jsonl
```

`replay` re-runs the same prompt with the model / temperature / seed recorded on
the trace's `run` span (calls Ollama again — it does not cassette responses).
Traces from before the `run` span was added cannot be replayed; run once more first.

`diff` compares prompt, knobs, tool path, tokens, latency, and final answer.

## Evals

From the repo root (Ollama must be running):

```bash
glassbox eval
# or selected cases:
glassbox eval calc_precedence sql_pets
python -m evals.run_evals
```
