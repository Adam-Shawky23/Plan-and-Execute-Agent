from dataclasses import dataclass


@dataclass
class Subtask:
    id: int
    description: str


@dataclass
class Plan:
    subtasks: list[Subtask]


@dataclass
class SubtaskResult:
    subtask_id: int
    description: str
    result: str
    incomplete: bool = False
