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
