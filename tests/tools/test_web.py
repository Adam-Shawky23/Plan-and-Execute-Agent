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
