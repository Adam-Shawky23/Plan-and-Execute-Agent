from html.parser import HTMLParser

import requests
from duckduckgo_search import DDGS


def web_search(query: str, max_results: int = 5) -> str:
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))

    if not results:
        return "No results found."

    lines = [
        f"- {r.get('title')}: {r.get('href')}\n  {r.get('body')}" for r in results
    ]
    return "\n".join(lines)


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        stripped = data.strip()
        if stripped:
            self.parts.append(stripped)


def fetch_url(url: str) -> str:
    response = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()

    parser = _TextExtractor()
    parser.feed(response.text)
    text = "\n".join(parser.parts)
    return text[:5000]
