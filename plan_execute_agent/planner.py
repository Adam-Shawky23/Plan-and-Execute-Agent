import json

from .models import Plan, Subtask

PLANNER_SYSTEM_PROMPT = (
    "You are a task planner. Break the user's task into a short, ordered list "
    "of concrete, independently-executable subtasks. Respond ONLY with JSON "
    'matching this schema: {"subtasks": [{"id": 1, "description": "..."}, ...]}. '
    "Use as few subtasks as the task genuinely needs."
)

PLANNER_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "subtasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "description": {"type": "string"},
                },
                "required": ["id", "description"],
            },
        }
    },
    "required": ["subtasks"],
}

RETRY_PROMPT = "Your last response was not valid JSON matching the schema. Respond with ONLY the JSON object."

MAX_ATTEMPTS = 2


class Planner:
    def __init__(self, client):
        self.client = client

    def plan(self, task: str) -> Plan:
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]

        for _ in range(MAX_ATTEMPTS):
            message = self.client.chat(messages, format=PLANNER_JSON_SCHEMA)
            parsed = self._try_parse(message.get("content", ""))
            if parsed is not None:
                return parsed
            messages.append({"role": "user", "content": RETRY_PROMPT})

        return Plan(subtasks=[Subtask(id=1, description=task)])

    def _try_parse(self, content: str) -> Plan | None:
        try:
            data = json.loads(content)
            subtasks = [
                Subtask(id=item["id"], description=item["description"])
                for item in data["subtasks"]
            ]
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

        return Plan(subtasks=subtasks) if subtasks else None
