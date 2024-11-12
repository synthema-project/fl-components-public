import pytest

from schemas.task import TaskRead, Status, Task


@pytest.fixture
def task():
    return TaskRead(
        user_id="user",
        model_name="model",
        model_version=1,
        num_global_iterations=10,
        use_case="use_case",
        id=1,
        created_at="2021-01-01T00:00:00",
        status="pending",
        run_name="run_name",
        experiment_name="experiment_name",
    )


def test_task_serde(task):
    js = task.model_dump_json()

    new_task = TaskRead.from_json(js)

    new_task_dict = new_task.model_dump()
    task_dict = new_task.model_dump()
    assert new_task_dict == task_dict


def test_task_change_status(task):
    task_db = Task.model_validate(task)

    task_db.change_status("RUNNING")
    assert task_db.status == Status.RUNNING
    task_db.change_status("cancelled")
    assert task_db.status == Status.CANCELLED
