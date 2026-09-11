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
