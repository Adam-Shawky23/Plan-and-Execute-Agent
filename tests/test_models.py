from plan_execute_agent.models import Plan, Subtask, SubtaskResult


def test_subtask_fields():
    s = Subtask(id=1, description="do a thing")
    assert s.id == 1
    assert s.description == "do a thing"


def test_plan_holds_subtasks():
    plan = Plan(subtasks=[Subtask(id=1, description="a"), Subtask(id=2, description="b")])
    assert len(plan.subtasks) == 2
    assert plan.subtasks[0].description == "a"


def test_subtask_result_defaults_not_incomplete():
    r = SubtaskResult(subtask_id=1, description="a", result="done")
    assert r.incomplete is False


def test_subtask_result_can_be_marked_incomplete():
    r = SubtaskResult(subtask_id=1, description="a", result="partial", incomplete=True)
    assert r.incomplete is True
