# Plan-and-Execute Agent

[![Tests](https://github.com/adamshawky/plan-execute-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/adamshawky/plan-execute-agent/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)

A standalone, local-only agent that takes a complex task, breaks it into an
ordered list of subtasks (planning), and executes each subtask using a mix
of LLM reasoning and tool calls (action) via a local [Ollama](https://ollama.com)
model. No cloud LLM API keys required.

## How it works

1. **Plan** — one Ollama call breaks your task into a short, ordered list of
   concrete subtasks.
2. **Execute** — each subtask runs its own reason -> tool-call -> observe
   loop (a "mini-ReAct" loop) using the built-in tools, until it produces a
   final answer for that subtask or hits its iteration cap.
3. **Synthesize** — one final Ollama call combines all subtask results into
   a single answer.

The plan is static (no replanning mid-run), and the whole run is fully
autonomous — no confirmation prompts.

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running (`ollama serve`)
- A tool-calling-capable model pulled, e.g.: `ollama pull qwen2.5`

## Install

```bash
pip install -e .
```

## Usage

```bash
plan-execute-agent "Summarize the README files in this repo and list any TODOs"
```

Options:

- `--model` — Ollama model to use (default: `qwen2.5`)
- `--host` — Ollama server URL (default: `http://localhost:11434`)

## Built-in tools

- `run_shell_command` — run a shell command
- `read_file` / `write_file` / `list_dir` — local filesystem access
- `web_search` — DuckDuckGo web search (no API key)
- `fetch_url` — fetch and extract text from a URL

Add a new tool by writing a function returning a string, then registering
it in `plan_execute_agent/tools/__init__.py::build_default_registry`.

> **Security note:** `run_shell_command` gives the model unrestricted shell
> access on your machine, and the agent runs fully autonomously with no
> confirmation prompts. Only point this at models/tasks you trust, and
> consider running it in a sandbox or container if the task is untrusted.

## Troubleshooting

- `Could not connect to Ollama...` — start the server with `ollama serve`.
- `Ollama request failed (404): model '...' not found` — pull the model
  first, e.g. `ollama pull qwen2.5`, or pass `--model <a model you have>`
  (see installed models with `ollama list`).

## Development

```bash
pip install -e .
pip install pytest
pytest -v
```

The suite includes one integration test (`test_full_run_against_real_ollama`
in `tests/test_integration.py`) that is always skipped in CI since it needs
a live Ollama server with the default model pulled. To run it locally,
remove its `@pytest.mark.skip` decorator and run `pytest -v`.

## Design docs

- Spec: `docs/specs/2026-09-11-plan-execute-agent-design.md`
- Plan: `docs/plans/2026-09-11-plan-execute-agent.md`

## License

MIT — see [LICENSE](LICENSE).
