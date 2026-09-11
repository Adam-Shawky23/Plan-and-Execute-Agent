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
