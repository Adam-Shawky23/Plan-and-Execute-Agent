import argparse
import sys

from .agent import Agent
from .executor import Executor
from .ollama_client import OllamaClient, OllamaConnectionError
from .planner import Planner
from .tools import build_default_registry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan-and-execute local agent.")
    parser.add_argument("task", help="The task to accomplish.")
    parser.add_argument("--model", default="qwen2.5", help="Ollama model to use.")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama host URL.")
    args = parser.parse_args(argv)

    client = OllamaClient(model=args.model, host=args.host)
    registry = build_default_registry()
    planner = Planner(client)
    executor = Executor(client, registry)
    agent = Agent(client, planner, executor)

    try:
        subtasks, results, final_answer = agent.run(args.task)
    except OllamaConnectionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Plan:")
    for subtask in subtasks:
        print(f"  {subtask.id}. {subtask.description}")
    print()

    for result in results:
        status = " (incomplete)" if result.incomplete else ""
        print(f"[Subtask {result.subtask_id}]{status} {result.description}")
        print(f"  -> {result.result}\n")

    print("Final answer:")
    print(final_answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
