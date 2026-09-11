from .models import SubtaskResult

SYNTHESIS_SYSTEM_PROMPT = (
    "You are given the original task and the results of each subtask that was "
    "executed to complete it. Write one clear final answer for the user, "
    "synthesizing all subtask results."
)


class Agent:
    def __init__(self, client, planner, executor):
        self.client = client
        self.planner = planner
        self.executor = executor

    def run(self, task: str):
        plan = self.planner.plan(task)

        results: list[SubtaskResult] = []
        for subtask in plan.subtasks:
            result = self.executor.run(subtask, results)
            results.append(result)

        results_text = "\n".join(
            f"Subtask {r.subtask_id} ({r.description}): {r.result}" for r in results
        )
        messages = [
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Original task: {task}\n\nSubtask results:\n{results_text}",
            },
        ]
        message = self.client.chat(messages)

        return plan.subtasks, results, message["content"]
