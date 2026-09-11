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
