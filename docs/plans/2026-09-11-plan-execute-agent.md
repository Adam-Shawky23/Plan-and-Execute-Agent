# Plan-and-Execute Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone Python CLI that plans a complex task into an ordered list of subtasks and executes each with a local-Ollama-driven mini-ReAct loop (reasoning + tool calls), then synthesizes a final answer.

**Architecture:** A `Planner` makes one structured-JSON Ollama call to produce a static ordered subtask list. An `Executor` runs each subtask through its own reason→tool-call→observe loop (capped iterations) against a shared `ToolRegistry` (shell, files, web search/fetch). An `Agent` orchestrates planner → executor-per-subtask → one final synthesis call. A thin `cli.py` wires it together and prints progress.

**Tech Stack:** Python 3.11+, `requests` (HTTP to Ollama and web fetch), `duckduckgo-search` (web search), `pytest` (tests), local Ollama server running `qwen2.5` (configurable).

**Spec:** `docs/superpowers/specs/2026-09-11-plan-execute-agent-design.md`

## Global Constraints

- Local Ollama only; default model `qwen2.5`, must be configurable (no model name hardcoded into logic).
- No external LLM API keys anywhere in the codebase.
- Tool registry must be extensible: adding a tool = adding one file + one `register()` call, no changes to `Executor`/`Agent`.
- Planning is static: one upfront plan, no replanning mid-run.
- Each subtask executes via its own mini-ReAct loop with a hard iteration cap (default 8).
- Fully autonomous: no interactive confirmation/approval prompts anywhere in the run path.
- In-memory only: no run state written to disk between invocations.
- Tool call exceptions must be caught and returned as an error string to the model, never raised into the loop.
- Web search uses DuckDuckGo with no API key.

---

### Task 1: Project Scaffolding & Core Data Models

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `plan_execute_agent/__init__.py`
- Create: `plan_execute_agent/models.py`
- Create: `tests/__init__.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces: `Subtask(id: int, description: str)`, `Plan(subtasks: list[Subtask])`, `SubtaskResult(subtask_id: int, description: str, result: str, incomplete: bool = False)` — all plain dataclasses, used by every later task.

- [ ] **Step 1: Create project scaffolding files**

`pyproject.toml`:
```toml
[project]
name = "plan-execute-agent"
version = "0.1.0"
description = "A local, Ollama-powered plan-and-execute agent."
requires-python = ">=3.11"
dependencies = [
    "requests>=2.31",
    "duckduckgo-search>=6.0",
]

[project.scripts]
plan-execute-agent = "plan_execute_agent.cli:main"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["plan_execute_agent*"]
```

`requirements.txt`:
```
requests>=2.31
duckduckgo-search>=6.0
pytest>=8.0
```

`plan_execute_agent/__init__.py`:
```python
```

`tests/__init__.py`:
```python
```

- [ ] **Step 2: Write the failing test for the data models**

`tests/test_models.py`:
```python
from plan_execute_agent.models import Plan, Subtask, SubtaskResult


def test_subtask_fields():
    s = Subtask(id=1, description="do a thing")
    assert s.id == 1
    assert s.description == "do a thing"


def test_plan_holds_subtasks():
    plan = Plan(subtasks=[Subtask(id=1, description="a"), Subtask(id=2, description="b")])
    assert len(plan.subtasks) == 2
    assert plan.subtasks[0].description == "a"


def test_subtask_result_defaults_not_incomplete():
    r = SubtaskResult(subtask_id=1, description="a", result="done")
    assert r.incomplete is False


def test_subtask_result_can_be_marked_incomplete():
    r = SubtaskResult(subtask_id=1, description="a", result="partial", incomplete=True)
    assert r.incomplete is True
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'plan_execute_agent.models'`

- [ ] **Step 4: Write minimal implementation**

`plan_execute_agent/models.py`:
```python
from dataclasses import dataclass


@dataclass
class Subtask:
    id: int
    description: str


@dataclass
class Plan:
    subtasks: list[Subtask]


@dataclass
class SubtaskResult:
    subtask_id: int
    description: str
    result: str
    incomplete: bool = False
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_models.py -v`
Expected: PASS (4 passed)

- [ ] **Step 6: Install the project in editable mode**

Run: `pip install -e ".[dev]" 2>/dev/null || pip install -e . && pip install pytest`

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml requirements.txt plan_execute_agent/__init__.py plan_execute_agent/models.py tests/__init__.py tests/test_models.py
git commit -m "Add project scaffolding and core data models"
```

---

### Task 2: Ollama Client Wrapper

**Files:**
- Create: `plan_execute_agent/ollama_client.py`
- Test: `tests/test_ollama_client.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `OllamaConnectionError(Exception)`; `OllamaClient(model: str = "qwen2.5", host: str = "http://localhost:11434")` with method `chat(messages: list[dict], tools: list[dict] | None = None, format: dict | None = None) -> dict` returning the response's `message` dict (a dict with at least `content: str` and optionally `tool_calls: list[dict]`).

- [ ] **Step 1: Write the failing tests**

`tests/test_ollama_client.py`:
```python
from unittest.mock import patch, MagicMock

import pytest
import requests

from plan_execute_agent.ollama_client import OllamaClient, OllamaConnectionError


def _mock_response(message):
    resp = MagicMock()
    resp.json.return_value = {"message": message}
    resp.raise_for_status.return_value = None
    return resp


@patch("plan_execute_agent.ollama_client.requests.post")
def test_chat_returns_message_dict(mock_post):
    mock_post.return_value = _mock_response({"content": "hello"})
    client = OllamaClient(model="qwen2.5", host="http://localhost:11434")

    result = client.chat([{"role": "user", "content": "hi"}])

    assert result == {"content": "hello"}
    called_payload = mock_post.call_args.kwargs["json"]
    assert called_payload["model"] == "qwen2.5"
    assert called_payload["stream"] is False
    assert "tools" not in called_payload
    assert "format" not in called_payload


@patch("plan_execute_agent.ollama_client.requests.post")
def test_chat_passes_tools_and_format(mock_post):
    mock_post.return_value = _mock_response({"content": "hi", "tool_calls": []})
    client = OllamaClient()

    tools = [{"type": "function", "function": {"name": "x"}}]
    fmt = {"type": "object"}
    client.chat([{"role": "user", "content": "hi"}], tools=tools, format=fmt)

    called_payload = mock_post.call_args.kwargs["json"]
    assert called_payload["tools"] == tools
    assert called_payload["format"] == fmt


@patch("plan_execute_agent.ollama_client.requests.post")
def test_chat_raises_ollama_connection_error_on_connection_failure(mock_post):
    mock_post.side_effect = requests.exceptions.ConnectionError()
    client = OllamaClient()

    with pytest.raises(OllamaConnectionError):
        client.chat([{"role": "user", "content": "hi"}])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ollama_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'plan_execute_agent.ollama_client'`

- [ ] **Step 3: Write minimal implementation**

`plan_execute_agent/ollama_client.py`:
```python
import requests


class OllamaConnectionError(Exception):
    pass


class OllamaClient:
    def __init__(self, model: str = "qwen2.5", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host.rstrip("/")

    def chat(self, messages, tools=None, format=None) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
        if format:
            payload["format"] = format

        try:
            response = requests.post(f"{self.host}/api/chat", json=payload, timeout=120)
        except requests.exceptions.ConnectionError as exc:
            raise OllamaConnectionError(
                f"Could not connect to Ollama at {self.host}. "
                f"Is `ollama serve` running and is the model pulled?"
            ) from exc

        response.raise_for_status()
        return response.json()["message"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ollama_client.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add plan_execute_agent/ollama_client.py tests/test_ollama_client.py
git commit -m "Add Ollama client wrapper with tool/format support"
```

---

### Task 3: Tool Registry

**Files:**
- Create: `plan_execute_agent/tools/__init__.py` (empty for now — populated in Task 6)
- Create: `plan_execute_agent/tools/registry.py`
- Create: `tests/tools/__init__.py`
- Test: `tests/tools/test_registry.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `ToolRegistry` with `register(name: str, description: str, parameters: dict, func: Callable[..., str]) -> None`, `schemas() -> list[dict]` (Ollama tool-schema format), `call(name: str, arguments: dict) -> str` (never raises — catches exceptions and returns an error string).

- [ ] **Step 1: Write the failing tests**

`tests/tools/__init__.py`:
```python
```

`tests/tools/test_registry.py`:
```python
from plan_execute_agent.tools.registry import ToolRegistry


def _params():
    return {
        "type": "object",
        "properties": {"x": {"type": "string"}},
        "required": ["x"],
    }


def test_schemas_reflects_registered_tools():
    registry = ToolRegistry()
    registry.register("echo", "Echoes input.", _params(), lambda x: x)

    schemas = registry.schemas()

    assert len(schemas) == 1
    assert schemas[0]["type"] == "function"
    assert schemas[0]["function"]["name"] == "echo"
    assert schemas[0]["function"]["description"] == "Echoes input."
    assert schemas[0]["function"]["parameters"] == _params()


def test_call_invokes_registered_function():
    registry = ToolRegistry()
    registry.register("echo", "Echoes input.", _params(), lambda x: f"got:{x}")

    result = registry.call("echo", {"x": "hi"})

    assert result == "got:hi"


def test_call_unknown_tool_returns_error_string():
    registry = ToolRegistry()

    result = registry.call("missing", {})

    assert "unknown tool" in result.lower()


def test_call_catches_exceptions_and_returns_error_string():
    def boom(x):
        raise ValueError("bad input")

    registry = ToolRegistry()
    registry.register("boom", "Always fails.", _params(), boom)

    result = registry.call("boom", {"x": "hi"})

    assert "bad input" in result
    assert "error" in result.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/tools/test_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'plan_execute_agent.tools'`

- [ ] **Step 3: Write minimal implementation**

`plan_execute_agent/tools/__init__.py`:
```python
```

`plan_execute_agent/tools/registry.py`:
```python
from dataclasses import dataclass
from typing import Callable


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    func: Callable[..., str]


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, name: str, description: str, parameters: dict, func: Callable[..., str]) -> None:
        self._tools[name] = Tool(name=name, description=description, parameters=parameters, func=func)

    def schemas(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]

    def call(self, name: str, arguments: dict) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: unknown tool '{name}'"
        try:
            return tool.func(**arguments)
        except Exception as exc:
            return f"Error running tool '{name}': {exc}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/tools/test_registry.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add plan_execute_agent/tools/__init__.py plan_execute_agent/tools/registry.py tests/tools/__init__.py tests/tools/test_registry.py
git commit -m "Add tool registry with schema generation and safe dispatch"
```

---

### Task 4: Shell & File Tools

**Files:**
- Create: `plan_execute_agent/tools/shell.py`
- Create: `plan_execute_agent/tools/files.py`
- Test: `tests/tools/test_shell.py`
- Test: `tests/tools/test_files.py`

**Interfaces:**
- Consumes: nothing (pure functions, no dependency on `ToolRegistry`).
- Produces: `run_shell_command(command: str) -> str`; `read_file(path: str) -> str`; `write_file(path: str, content: str) -> str`; `list_dir(path: str = ".") -> str`. All raise on failure (caught later by `ToolRegistry.call`, not here).

- [ ] **Step 1: Write the failing test for the shell tool**

`tests/tools/test_shell.py`:
```python
from unittest.mock import patch, MagicMock

from plan_execute_agent.tools.shell import run_shell_command


@patch("plan_execute_agent.tools.shell.subprocess.run")
def test_run_shell_command_includes_stdout_and_exit_code(mock_run):
    mock_run.return_value = MagicMock(stdout="hello\n", stderr="", returncode=0)

    result = run_shell_command("echo hello")

    assert "hello" in result
    assert "[exit code: 0]" in result


@patch("plan_execute_agent.tools.shell.subprocess.run")
def test_run_shell_command_includes_stderr_when_present(mock_run):
    mock_run.return_value = MagicMock(stdout="", stderr="boom", returncode=1)

    result = run_shell_command("false")

    assert "boom" in result
    assert "[exit code: 1]" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/tools/test_shell.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'plan_execute_agent.tools.shell'`

- [ ] **Step 3: Write minimal implementation for the shell tool**

`plan_execute_agent/tools/shell.py`:
```python
import subprocess


def run_shell_command(command: str) -> str:
    result = subprocess.run(
        command, shell=True, capture_output=True, text=True, timeout=30
    )
    output = result.stdout or ""
    if result.stderr:
        output += f"\n[stderr]\n{result.stderr}"
    output += f"\n[exit code: {result.returncode}]"
    return output
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/tools/test_shell.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Write the failing tests for the file tools**

`tests/tools/test_files.py`:
```python
from plan_execute_agent.tools.files import read_file, write_file, list_dir


def test_write_then_read_file(tmp_path):
    file_path = tmp_path / "note.txt"

    write_result = write_file(str(file_path), "hello world")
    read_result = read_file(str(file_path))

    assert "11 characters" in write_result
    assert read_result == "hello world"


def test_list_dir_shows_files_and_dirs(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "subdir").mkdir()

    result = list_dir(str(tmp_path))

    assert "a.txt" in result
    assert "subdir/" in result


def test_list_dir_empty_directory(tmp_path):
    result = list_dir(str(tmp_path))

    assert result == "(empty directory)"
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `pytest tests/tools/test_files.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'plan_execute_agent.tools.files'`

- [ ] **Step 7: Write minimal implementation for the file tools**

`plan_execute_agent/tools/files.py`:
```python
from pathlib import Path


def read_file(path: str) -> str:
    return Path(path).read_text()


def write_file(path: str, content: str) -> str:
    Path(path).write_text(content)
    return f"Wrote {len(content)} characters to {path}"


def list_dir(path: str = ".") -> str:
    entries = sorted(
        p.name + ("/" if p.is_dir() else "") for p in Path(path).iterdir()
    )
    return "\n".join(entries) if entries else "(empty directory)"
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `pytest tests/tools/test_files.py -v`
Expected: PASS (3 passed)

- [ ] **Step 9: Commit**

```bash
git add plan_execute_agent/tools/shell.py plan_execute_agent/tools/files.py tests/tools/test_shell.py tests/tools/test_files.py
git commit -m "Add shell and file tools"
```

---

### Task 5: Web Tools (search & fetch)

**Files:**
- Create: `plan_execute_agent/tools/web.py`
- Test: `tests/tools/test_web.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (pure functions).
- Produces: `web_search(query: str, max_results: int = 5) -> str`; `fetch_url(url: str) -> str`.

- [ ] **Step 1: Write the failing tests**

`tests/tools/test_web.py`:
```python
from unittest.mock import patch, MagicMock

from plan_execute_agent.tools.web import web_search, fetch_url


@patch("plan_execute_agent.tools.web.DDGS")
def test_web_search_formats_results(mock_ddgs_cls):
    mock_ddgs = MagicMock()
    mock_ddgs.__enter__.return_value = mock_ddgs
    mock_ddgs.text.return_value = [
        {"title": "Example", "href": "https://example.com", "body": "An example site."}
    ]
    mock_ddgs_cls.return_value = mock_ddgs

    result = web_search("example query")

    assert "Example" in result
    assert "https://example.com" in result
    assert "An example site." in result


@patch("plan_execute_agent.tools.web.DDGS")
def test_web_search_no_results(mock_ddgs_cls):
    mock_ddgs = MagicMock()
    mock_ddgs.__enter__.return_value = mock_ddgs
    mock_ddgs.text.return_value = []
    mock_ddgs_cls.return_value = mock_ddgs

    result = web_search("nothing matches this")

    assert result == "No results found."


@patch("plan_execute_agent.tools.web.requests.get")
def test_fetch_url_extracts_text(mock_get):
    mock_response = MagicMock()
    mock_response.text = "<html><body><h1>Title</h1><p>Some text.</p></body></html>"
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    result = fetch_url("https://example.com")

    assert "Title" in result
    assert "Some text." in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/tools/test_web.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'plan_execute_agent.tools.web'`

- [ ] **Step 3: Write minimal implementation**

`plan_execute_agent/tools/web.py`:
```python
from html.parser import HTMLParser

import requests
from duckduckgo_search import DDGS


def web_search(query: str, max_results: int = 5) -> str:
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))

    if not results:
        return "No results found."

    lines = [
        f"- {r.get('title')}: {r.get('href')}\n  {r.get('body')}" for r in results
    ]
    return "\n".join(lines)


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        stripped = data.strip()
        if stripped:
            self.parts.append(stripped)


def fetch_url(url: str) -> str:
    response = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()

    parser = _TextExtractor()
    parser.feed(response.text)
    text = "\n".join(parser.parts)
    return text[:5000]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/tools/test_web.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add plan_execute_agent/tools/web.py tests/tools/test_web.py
git commit -m "Add web search and fetch tools using DuckDuckGo"
```

---

### Task 6: Default Tool Registry Wiring

**Files:**
- Modify: `plan_execute_agent/tools/__init__.py`
- Test: `tests/tools/test_default_registry.py`

**Interfaces:**
- Consumes: `ToolRegistry` (Task 3), `shell.run_shell_command`, `files.{read_file,write_file,list_dir}` (Task 4), `web.{web_search,fetch_url}` (Task 5).
- Produces: `build_default_registry() -> ToolRegistry` pre-populated with all six tools, used by `cli.py` in Task 10.

- [ ] **Step 1: Write the failing test**

`tests/tools/test_default_registry.py`:
```python
from plan_execute_agent.tools import build_default_registry


def test_default_registry_contains_all_expected_tools():
    registry = build_default_registry()

    names = {schema["function"]["name"] for schema in registry.schemas()}

    assert names == {
        "run_shell_command",
        "read_file",
        "write_file",
        "list_dir",
        "web_search",
        "fetch_url",
    }


def test_default_registry_tools_are_callable():
    registry = build_default_registry()

    result = registry.call("list_dir", {"path": "."})

    assert isinstance(result, str)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/tools/test_default_registry.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_default_registry'`

- [ ] **Step 3: Write minimal implementation**

`plan_execute_agent/tools/__init__.py`:
```python
from .registry import ToolRegistry
from . import shell, files, web


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()

    registry.register(
        name="run_shell_command",
        description="Run a shell command and return its stdout, stderr, and exit code.",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to run."},
            },
            "required": ["command"],
        },
        func=shell.run_shell_command,
    )
    registry.register(
        name="read_file",
        description="Read and return the contents of a text file.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read."},
            },
            "required": ["path"],
        },
        func=files.read_file,
    )
    registry.register(
        name="write_file",
        description="Write text content to a file, overwriting it if it already exists.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to write."},
                "content": {"type": "string", "description": "Content to write to the file."},
            },
            "required": ["path", "content"],
        },
        func=files.write_file,
    )
    registry.register(
        name="list_dir",
        description="List the contents of a directory.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path.", "default": "."},
            },
            "required": [],
        },
        func=files.list_dir,
    )
    registry.register(
        name="web_search",
        description="Search the web via DuckDuckGo and return matching results.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
                "max_results": {"type": "integer", "description": "Maximum results to return.", "default": 5},
            },
            "required": ["query"],
        },
        func=web.web_search,
    )
    registry.register(
        name="fetch_url",
        description="Fetch a URL and return its extracted text content.",
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to fetch."},
            },
            "required": ["url"],
        },
        func=web.fetch_url,
    )

    return registry
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/tools/test_default_registry.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add plan_execute_agent/tools/__init__.py tests/tools/test_default_registry.py
git commit -m "Wire built-in tools into a default tool registry"
```

---

### Task 7: Planner

**Files:**
- Create: `plan_execute_agent/planner.py`
- Test: `tests/test_planner.py`

**Interfaces:**
- Consumes: `Plan`, `Subtask` (Task 1, `plan_execute_agent.models`); any object with a `.chat(messages, tools=None, format=None) -> dict` method matching `OllamaClient` (Task 2) — tests use a fake, not the real client.
- Produces: `Planner(client)` with `plan(task: str) -> Plan`, used by `Agent` in Task 9.

- [ ] **Step 1: Write the failing tests**

`tests/test_planner.py`:
```python
import json

from plan_execute_agent.planner import Planner


class FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def chat(self, messages, tools=None, format=None):
        self.calls.append(messages)
        return self._responses.pop(0)


def test_plan_parses_valid_json_response():
    valid = {"content": json.dumps({"subtasks": [{"id": 1, "description": "step one"}, {"id": 2, "description": "step two"}]})}
    client = FakeClient([valid])
    planner = Planner(client)

    plan = planner.plan("do something complex")

    assert [s.description for s in plan.subtasks] == ["step one", "step two"]
    assert len(client.calls) == 1


def test_plan_retries_once_on_malformed_json_then_succeeds():
    invalid = {"content": "not json at all"}
    valid = {"content": json.dumps({"subtasks": [{"id": 1, "description": "step one"}]})}
    client = FakeClient([invalid, valid])
    planner = Planner(client)

    plan = planner.plan("do something")

    assert len(plan.subtasks) == 1
    assert len(client.calls) == 2


def test_plan_falls_back_to_single_subtask_after_two_failures():
    invalid = {"content": "still not json"}
    client = FakeClient([invalid, invalid])
    planner = Planner(client)

    plan = planner.plan("original task text")

    assert len(plan.subtasks) == 1
    assert plan.subtasks[0].description == "original task text"
    assert len(client.calls) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_planner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'plan_execute_agent.planner'`

- [ ] **Step 3: Write minimal implementation**

`plan_execute_agent/planner.py`:
```python
import json

from .models import Plan, Subtask

PLANNER_SYSTEM_PROMPT = (
    "You are a task planner. Break the user's task into a short, ordered list "
    "of concrete, independently-executable subtasks. Respond ONLY with JSON "
    'matching this schema: {"subtasks": [{"id": 1, "description": "..."}, ...]}. '
    "Use as few subtasks as the task genuinely needs."
)

PLANNER_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "subtasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "description": {"type": "string"},
                },
                "required": ["id", "description"],
            },
        }
    },
    "required": ["subtasks"],
}

RETRY_PROMPT = "Your last response was not valid JSON matching the schema. Respond with ONLY the JSON object."

MAX_ATTEMPTS = 2


class Planner:
    def __init__(self, client):
        self.client = client

    def plan(self, task: str) -> Plan:
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]

        for _ in range(MAX_ATTEMPTS):
            message = self.client.chat(messages, format=PLANNER_JSON_SCHEMA)
            parsed = self._try_parse(message.get("content", ""))
            if parsed is not None:
                return parsed
            messages.append({"role": "user", "content": RETRY_PROMPT})

        return Plan(subtasks=[Subtask(id=1, description=task)])

    def _try_parse(self, content: str) -> Plan | None:
        try:
            data = json.loads(content)
            subtasks = [
                Subtask(id=item["id"], description=item["description"])
                for item in data["subtasks"]
            ]
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

        return Plan(subtasks=subtasks) if subtasks else None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_planner.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add plan_execute_agent/planner.py tests/test_planner.py
git commit -m "Add planner with structured-JSON parsing and fallback"
```

---

### Task 8: Executor (mini-ReAct loop)

**Files:**
- Create: `plan_execute_agent/executor.py`
- Test: `tests/test_executor.py`

**Interfaces:**
- Consumes: `Subtask`, `SubtaskResult` (Task 1); any object with `.chat(messages, tools=None, format=None) -> dict` (Task 2, faked in tests); any object with `.schemas() -> list[dict]` and `.call(name, arguments) -> str` (Task 3/6, faked in tests).
- Produces: `Executor(client, registry, max_iterations: int = 8)` with `run(subtask: Subtask, context: list[SubtaskResult]) -> SubtaskResult`, used by `Agent` in Task 9.

- [ ] **Step 1: Write the failing tests**

`tests/test_executor.py`:
```python
from plan_execute_agent.executor import Executor
from plan_execute_agent.models import Subtask, SubtaskResult


class FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def chat(self, messages, tools=None, format=None):
        self.calls.append(messages)
        return self._responses.pop(0)


class FakeRegistry:
    def __init__(self):
        self.call_log = []

    def schemas(self):
        return [{"type": "function", "function": {"name": "noop"}}]

    def call(self, name, arguments):
        self.call_log.append((name, arguments))
        return "tool observation"


def test_run_returns_immediate_text_answer():
    client = FakeClient([{"content": "the answer"}])
    registry = FakeRegistry()
    executor = Executor(client, registry)

    result = executor.run(Subtask(id=1, description="do X"), context=[])

    assert result == SubtaskResult(subtask_id=1, description="do X", result="the answer")
    assert registry.call_log == []


def test_run_executes_tool_call_then_returns_text_answer():
    client = FakeClient([
        {"content": "", "tool_calls": [{"function": {"name": "noop", "arguments": {"a": 1}}}]},
        {"content": "final answer after tool use"},
    ])
    registry = FakeRegistry()
    executor = Executor(client, registry)

    result = executor.run(Subtask(id=2, description="do Y"), context=[])

    assert result.result == "final answer after tool use"
    assert registry.call_log == [("noop", {"a": 1})]


def test_run_hits_iteration_cap_and_marks_incomplete():
    tool_call_message = {"content": "", "tool_calls": [{"function": {"name": "noop", "arguments": {}}}]}
    responses = [tool_call_message] * 8 + [{"content": "partial summary"}]
    client = FakeClient(responses)
    registry = FakeRegistry()
    executor = Executor(client, registry, max_iterations=8)

    result = executor.run(Subtask(id=3, description="do Z"), context=[])

    assert result.incomplete is True
    assert result.result == "partial summary"
    assert len(registry.call_log) == 8


def test_run_passes_prior_context_into_prompt():
    client = FakeClient([{"content": "done"}])
    registry = FakeRegistry()
    executor = Executor(client, registry)
    prior = [SubtaskResult(subtask_id=1, description="first", result="first result")]

    executor.run(Subtask(id=2, description="second"), context=prior)

    first_call_messages = client.calls[0]
    user_message = next(m["content"] for m in first_call_messages if m["role"] == "user")
    assert "first result" in user_message
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_executor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'plan_execute_agent.executor'`

- [ ] **Step 3: Write minimal implementation**

`plan_execute_agent/executor.py`:
```python
from .models import SubtaskResult, Subtask

EXECUTOR_SYSTEM_PROMPT = (
    "You are executing one subtask of a larger plan. Use the available tools "
    "as needed. When you have fully completed the subtask, respond with plain "
    "text (no tool call) giving the result."
)

SUMMARIZE_PROMPT = (
    "Iteration limit reached. Summarize what you have accomplished so far for this subtask."
)


class Executor:
    def __init__(self, client, registry, max_iterations: int = 8):
        self.client = client
        self.registry = registry
        self.max_iterations = max_iterations

    def run(self, subtask: Subtask, context: list[SubtaskResult]) -> SubtaskResult:
        context_text = "\n".join(
            f"Subtask {r.subtask_id} ({r.description}): {r.result}" for r in context
        ) or "(none)"

        messages = [
            {"role": "system", "content": EXECUTOR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Prior results:\n{context_text}\n\nYour subtask: {subtask.description}",
            },
        ]
        tools = self.registry.schemas()

        for _ in range(self.max_iterations):
            message = self.client.chat(messages, tools=tools)
            tool_calls = message.get("tool_calls")

            if not tool_calls:
                return SubtaskResult(
                    subtask_id=subtask.id,
                    description=subtask.description,
                    result=message["content"],
                )

            messages.append(message)
            for call in tool_calls:
                name = call["function"]["name"]
                arguments = call["function"]["arguments"]
                observation = self.registry.call(name, arguments)
                messages.append({"role": "tool", "content": observation})

        messages.append({"role": "user", "content": SUMMARIZE_PROMPT})
        message = self.client.chat(messages)
        return SubtaskResult(
            subtask_id=subtask.id,
            description=subtask.description,
            result=message["content"],
            incomplete=True,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_executor.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add plan_execute_agent/executor.py tests/test_executor.py
git commit -m "Add executor with mini-ReAct loop and iteration cap"
```

---

### Task 9: Agent (orchestration + synthesis)

**Files:**
- Create: `plan_execute_agent/agent.py`
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: `Plan`, `Subtask`, `SubtaskResult` (Task 1); any object with `.chat(messages, tools=None, format=None) -> dict` (Task 2, faked); any object with `.plan(task: str) -> Plan` (Task 7, faked); any object with `.run(subtask, context) -> SubtaskResult` (Task 8, faked).
- Produces: `Agent(client, planner, executor)` with `run(task: str) -> tuple[list[Subtask], list[SubtaskResult], str]`, used by `cli.py` in Task 10.

- [ ] **Step 1: Write the failing tests**

`tests/test_agent.py`:
```python
from plan_execute_agent.agent import Agent
from plan_execute_agent.models import Plan, Subtask, SubtaskResult


class FakeClient:
    def __init__(self, response):
        self._response = response
        self.calls = []

    def chat(self, messages, tools=None, format=None):
        self.calls.append(messages)
        return self._response


class FakePlanner:
    def __init__(self, plan):
        self._plan = plan

    def plan(self, task):
        return self._plan


class FakeExecutor:
    def __init__(self):
        self.run_calls = []

    def run(self, subtask, context):
        self.run_calls.append((subtask, list(context)))
        return SubtaskResult(subtask_id=subtask.id, description=subtask.description, result=f"result for {subtask.id}")


def test_agent_runs_each_subtask_in_order_with_growing_context():
    plan = Plan(subtasks=[Subtask(id=1, description="a"), Subtask(id=2, description="b")])
    client = FakeClient({"content": "final synthesized answer"})
    planner = FakePlanner(plan)
    executor = FakeExecutor()
    agent = Agent(client, planner, executor)

    subtasks, results, final_answer = agent.run("do the whole thing")

    assert [s.id for s in subtasks] == [1, 2]
    assert [r.result for r in results] == ["result for 1", "result for 2"]
    assert final_answer == "final synthesized answer"
    assert executor.run_calls[0][1] == []
    assert executor.run_calls[1][1] == [results[0]]


def test_agent_synthesis_call_includes_task_and_results():
    plan = Plan(subtasks=[Subtask(id=1, description="only step")])
    client = FakeClient({"content": "done"})
    planner = FakePlanner(plan)
    executor = FakeExecutor()
    agent = Agent(client, planner, executor)

    agent.run("original task text")

    synthesis_messages = client.calls[0]
    user_content = next(m["content"] for m in synthesis_messages if m["role"] == "user")
    assert "original task text" in user_content
    assert "result for 1" in user_content
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_agent.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'plan_execute_agent.agent'`

- [ ] **Step 3: Write minimal implementation**

`plan_execute_agent/agent.py`:
```python
from .models import SubtaskResult

SYNTHESIS_SYSTEM_PROMPT = (
    "You are given the original task and the results of each subtask that was "
    "executed to complete it. Write one clear final answer for the user, "
    "synthesizing all subtask results."
)


class Agent:
    def __init__(self, client, planner, executor):
        self.client = client
        self.planner = planner
        self.executor = executor

    def run(self, task: str):
        plan = self.planner.plan(task)

        results: list[SubtaskResult] = []
        for subtask in plan.subtasks:
            result = self.executor.run(subtask, results)
            results.append(result)

        results_text = "\n".join(
            f"Subtask {r.subtask_id} ({r.description}): {r.result}" for r in results
        )
        messages = [
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Original task: {task}\n\nSubtask results:\n{results_text}",
            },
        ]
        message = self.client.chat(messages)

        return plan.subtasks, results, message["content"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_agent.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add plan_execute_agent/agent.py tests/test_agent.py
git commit -m "Add agent orchestration with final synthesis call"
```

---

### Task 10: CLI Entrypoint

**Files:**
- Create: `plan_execute_agent/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `Agent` (Task 9), `Planner` (Task 7), `Executor` (Task 8), `OllamaClient`/`OllamaConnectionError` (Task 2), `build_default_registry` (Task 6).
- Produces: `main(argv: list[str] | None = None) -> int`, the `plan-execute-agent` console entrypoint declared in `pyproject.toml` (Task 1).

- [ ] **Step 1: Write the failing tests**

`tests/test_cli.py`:
```python
from unittest.mock import patch

from plan_execute_agent import cli
from plan_execute_agent.models import Subtask, SubtaskResult
from plan_execute_agent.ollama_client import OllamaConnectionError


class FakeAgent:
    def __init__(self, *args, **kwargs):
        pass

    def run(self, task):
        subtasks = [Subtask(id=1, description="step one")]
        results = [SubtaskResult(subtask_id=1, description="step one", result="did it")]
        return subtasks, results, "final answer text"


class FailingAgent:
    def __init__(self, *args, **kwargs):
        pass

    def run(self, task):
        raise OllamaConnectionError("Could not connect to Ollama at http://localhost:11434.")


@patch("plan_execute_agent.cli.Agent", FakeAgent)
def test_main_prints_plan_results_and_final_answer(capsys):
    exit_code = cli.main(["do something useful"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "step one" in captured.out
    assert "did it" in captured.out
    assert "final answer text" in captured.out


@patch("plan_execute_agent.cli.Agent", FailingAgent)
def test_main_reports_ollama_connection_error(capsys):
    exit_code = cli.main(["do something useful"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Could not connect to Ollama" in captured.err
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'plan_execute_agent.cli'`

- [ ] **Step 3: Write minimal implementation**

`plan_execute_agent/cli.py`:
```python
import argparse
import sys

from .agent import Agent
from .executor import Executor
from .ollama_client import OllamaClient, OllamaConnectionError
from .planner import Planner
from .tools import build_default_registry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan-and-execute local agent.")
    parser.add_argument("task", help="The task to accomplish.")
    parser.add_argument("--model", default="qwen2.5", help="Ollama model to use.")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama host URL.")
    args = parser.parse_args(argv)

    client = OllamaClient(model=args.model, host=args.host)
    registry = build_default_registry()
    planner = Planner(client)
    executor = Executor(client, registry)
    agent = Agent(client, planner, executor)

    try:
        subtasks, results, final_answer = agent.run(args.task)
    except OllamaConnectionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Plan:")
    for subtask in subtasks:
        print(f"  {subtask.id}. {subtask.description}")
    print()

    for result in results:
        status = " (incomplete)" if result.incomplete else ""
        print(f"[Subtask {result.subtask_id}]{status} {result.description}")
        print(f"  -> {result.result}\n")

    print("Final answer:")
    print(final_answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cli.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the full test suite**

Run: `pytest -v`
Expected: All tests across every task pass (models, ollama_client, tools/registry, tools/shell, tools/files, tools/web, tools/default_registry, planner, executor, agent, cli).

- [ ] **Step 6: Commit**

```bash
git add plan_execute_agent/cli.py tests/test_cli.py
git commit -m "Add CLI entrypoint wiring planner, executor, and agent together"
```

---

### Task 11: README & Manual Verification

**Files:**
- Create: `README.md`
- Create: `.gitignore`

**Interfaces:**
- Consumes: nothing (documentation task).
- Produces: nothing consumed by other tasks — this is the terminal documentation/verification task.

- [ ] **Step 1: Write `.gitignore`**

`.gitignore`:
```
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
.venv/
venv/
```

- [ ] **Step 2: Write `README.md`**

`README.md`:
```markdown
# Plan-and-Execute Agent

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

## Development

```bash
pip install -e .
pip install pytest
pytest -v
```

## Design docs

- Spec: `docs/superpowers/specs/2026-09-11-plan-execute-agent-design.md`
- Plan: `docs/superpowers/plans/2026-09-11-plan-execute-agent.md`
```

- [ ] **Step 3: Add a skip-by-default integration smoke test**

`tests/test_integration.py`:
```python
import pytest

from plan_execute_agent.agent import Agent
from plan_execute_agent.executor import Executor
from plan_execute_agent.ollama_client import OllamaClient
from plan_execute_agent.planner import Planner
from plan_execute_agent.tools import build_default_registry


@pytest.mark.skip(reason="requires a running local Ollama instance with qwen2.5 pulled")
def test_full_run_against_real_ollama():
    client = OllamaClient(model="qwen2.5")
    registry = build_default_registry()
    planner = Planner(client)
    executor = Executor(client, registry)
    agent = Agent(client, planner, executor)

    subtasks, results, final_answer = agent.run(
        "List the files in the current directory and count how many are Python files."
    )

    assert len(subtasks) >= 1
    assert len(results) == len(subtasks)
    assert isinstance(final_answer, str) and final_answer
```

- [ ] **Step 4: Run the full suite to confirm the new test is collected and skipped**

Run: `pytest -v`
Expected: all previous tests still pass; `test_full_run_against_real_ollama` shows as `SKIPPED`.

- [ ] **Step 5: Manually verify end-to-end with a real local Ollama instance**

Run:
```bash
ollama pull qwen2.5
ollama serve &
plan-execute-agent "List the files in the current directory and tell me how many Python files there are"
```
Expected: prints a plan, per-subtask progress showing `list_dir`/shell tool calls, and a final answer stating the Python file count.

- [ ] **Step 6: Commit**

```bash
git add README.md .gitignore tests/test_integration.py
git commit -m "Add README, gitignore, and skip-by-default integration smoke test"
```
