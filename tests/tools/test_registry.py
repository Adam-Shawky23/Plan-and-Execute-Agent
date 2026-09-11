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
