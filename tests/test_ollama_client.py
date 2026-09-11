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
