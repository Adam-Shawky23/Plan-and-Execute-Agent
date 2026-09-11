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
