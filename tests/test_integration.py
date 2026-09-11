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
