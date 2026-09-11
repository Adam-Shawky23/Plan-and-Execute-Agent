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
