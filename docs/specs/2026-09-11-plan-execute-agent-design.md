# Plan-and-Execute Agent — Design Spec

Date: 2026-09-11

## Purpose

A standalone Python CLI application that takes a complex user task, breaks
it into an ordered list of small subtasks (planning), and executes each
subtask autonomously using a mix of LLM reasoning and tool calls (action),
finally synthesizing all subtask results into one answer for the user.

## Goals / Constraints

- Runs entirely against a **local** LLM via **Ollama** (default model:
  `qwen2.5`, configurable — no model is hardcoded into logic).
- No external LLM API keys required.
- Tool set: shell commands, file read/write/list, web search (DuckDuckGo,
  no API key) and URL fetch. Tool registry is extensible — adding a new
  tool means adding one file, not touching the core loop.
- Planning is **static**: the planner produces the full ordered subtask
  list once, upfront. No replanning mid-run.
- Each subtask runs its own **mini-ReAct loop** (reason → tool call →
  observe, repeated until the model returns a plain-text final answer for
  that subtask, or an iteration cap is hit).
- Fully **autonomous** — no human-in-the-loop checkpoints or approval
  gates during a run.
- **In-memory only** — no persistence of run state to disk between
  invocations. Each CLI invocation is a single self-contained run.

## Non-Goals

- No replanning / dynamic plan revision.
- No human approval steps or interactive confirmation prompts.
- No run persistence, resumability, or history across invocations.
- No support for remote/hosted LLM providers (out of scope for this spec;
  the design should not actively prevent adding one later, but nothing is
  built for it now).

## Architecture

```
plan_execute_agent/
├── agent.py            # Agent: orchestrates planner -> executor loop -> final synthesis
├── planner.py          # Planner: task -> ordered list of subtasks
├── executor.py         # Executor: runs mini-ReAct loop for one subtask
├── tools/
│   ├── registry.py     # ToolRegistry: name -> {schema, callable}
│   ├── shell.py        # run_shell_command
│   ├── files.py        # read_file, write_file, list_dir
│   └── web.py          # web_search (DuckDuckGo), fetch_url
├── ollama_client.py    # thin wrapper around Ollama's /api/chat with tool-calling
├── cli.py              # `python -m plan_execute_agent "do X"` entrypoint
└── models.py           # dataclasses: Task, Subtask, SubtaskResult, Plan
```

## Components

- **`Planner`** — one Ollama call (tools disabled) with a system prompt
  instructing it to break the task into a numbered list of concrete,
  independently-executable subtasks, returned as structured JSON
  (`{subtasks: [{id, description}]}`) using Ollama's structured-output
  (`format: json`/schema) support for reliable parsing.
- **`ToolRegistry`** — holds tool schemas (name, description, JSON-schema
  parameters) and the corresponding Python callables. Built once at
  startup from `tools/*.py`. Its schema list is passed to the `Executor`
  for real tool-calling; the `Planner` does not call tools.
- **`Executor`** — given one `Subtask`, the `ToolRegistry`, and running
  context (prior subtasks' results), loops: call Ollama with tools
  enabled → if the response is a tool call, execute it via the registry
  and feed the result back as a tool message → repeat → when the model
  returns plain text (no tool call), that text is the subtask's final
  answer. Hard iteration cap per subtask (default 8) to prevent runaway
  loops.
- **`Agent`** — top-level orchestration: `plan = Planner.plan(task)`,
  then for each `Subtask` in order, `result = Executor.run(subtask,
  context)`, appending each `SubtaskResult` to a shared context visible to
  later subtasks. After all subtasks complete, one final Ollama call
  synthesizes all `SubtaskResult`s into the user-facing final answer.
- **`cli.py`** — takes the task as a CLI argument, prints the plan once
  generated, streams per-subtask progress (description → tool calls made
  → result), then prints the final synthesized answer.

## Data Flow

```
user task (string)
   -> Planner -> Plan (ordered Subtask[])
   -> for each Subtask:
        Executor mini-ReAct loop (reason -> tool call -> observe)*  -> SubtaskResult
        SubtaskResult appended to shared context (visible to later subtasks)
   -> Agent synthesizes all SubtaskResults -> final answer (printed to user)
```

## Error Handling

- Tool call exceptions are caught inside the registry wrapper, formatted
  as an error string, and fed back to the model as the tool result
  (never raised into the loop) — lets the model retry or adjust instead
  of crashing the run.
- Malformed planner JSON: retry once with a stricter prompt reminder; if
  still malformed, fall back to a single-subtask plan containing the
  original task verbatim.
- Ollama unreachable: fail fast with a clear error message telling the
  user to start Ollama, rather than retrying silently.
- Executor hits its iteration cap without a plain-text final answer:
  force one more call asking the model to summarize what it has so far;
  use that as the `SubtaskResult`, flagged `incomplete` in the shared
  context so later subtasks and the final synthesis know it may be
  partial.

## Testing

- Pytest unit tests per tool (`shell`, `files`, `web`) with
  subprocess/filesystem/network mocked.
- Unit tests for `ToolRegistry` schema generation.
- `Planner` and `Executor` tested against a scripted fake Ollama client
  (no real server required) — covers JSON parsing/fallback behavior,
  loop termination, the iteration cap, and context threading between
  subtasks.
- One optional integration smoke test, marked to skip by default, that
  exercises a real local Ollama instance end-to-end.

## Open Questions / Future Extensions (explicitly out of scope now)

- Swapping in remote LLM providers.
- Replanning/dynamic plan revision.
- Human-in-the-loop approval gates.
- Persisting/resuming runs.
