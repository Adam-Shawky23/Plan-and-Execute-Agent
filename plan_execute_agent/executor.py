from .models import SubtaskResult, Subtask

EXECUTOR_SYSTEM_PROMPT = (
    "You are executing one subtask of a larger plan. Use the available tools "
    "as needed. When you have fully completed the subtask, respond with plain "
    "text (no tool call) giving the result."
)

SUMMARIZE_PROMPT = (
    "Iteration limit reached. Summarize what you have accomplished so far for this subtask."
)


class Executor:
    def __init__(self, client, registry, max_iterations: int = 8):
        self.client = client
        self.registry = registry
        self.max_iterations = max_iterations

    def run(self, subtask: Subtask, context: list[SubtaskResult]) -> SubtaskResult:
        context_text = "\n".join(
            f"Subtask {r.subtask_id} ({r.description}): {r.result}" for r in context
        ) or "(none)"

        messages = [
            {"role": "system", "content": EXECUTOR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Prior results:\n{context_text}\n\nYour subtask: {subtask.description}",
            },
        ]
        tools = self.registry.schemas()

        for _ in range(self.max_iterations):
            message = self.client.chat(messages, tools=tools)
            tool_calls = message.get("tool_calls")

            if not tool_calls:
                return SubtaskResult(
                    subtask_id=subtask.id,
                    description=subtask.description,
                    result=message["content"],
                )

            messages.append(message)
            for call in tool_calls:
                name = call["function"]["name"]
                arguments = call["function"]["arguments"]
                observation = self.registry.call(name, arguments)
                messages.append({"role": "tool", "content": observation})

        messages.append({"role": "user", "content": SUMMARIZE_PROMPT})
        message = self.client.chat(messages)
        return SubtaskResult(
            subtask_id=subtask.id,
            description=subtask.description,
            result=message["content"],
            incomplete=True,
        )
