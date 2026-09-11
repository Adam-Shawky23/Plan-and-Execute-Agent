import requests


class OllamaConnectionError(Exception):
    pass


class OllamaClient:
    def __init__(self, model: str = "qwen2.5", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host.rstrip("/")

    def chat(self, messages, tools=None, format=None) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
        if format:
            payload["format"] = format

        try:
            response = requests.post(f"{self.host}/api/chat", json=payload, timeout=120)
        except requests.exceptions.ConnectionError as exc:
            raise OllamaConnectionError(
                f"Could not connect to Ollama at {self.host}. "
                f"Is `ollama serve` running and is the model pulled?"
            ) from exc

        response.raise_for_status()
        return response.json()["message"]
