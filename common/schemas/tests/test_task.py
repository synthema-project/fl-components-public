import pytest

from schemas.task import TaskRead, Status, Task


@pytest.fixture
def task():
    return TaskRead(
        user_id="user",
        model_uri="model",
        num_rounds=10,
        dataset_uris=["uri1", "uri2"],
        id=1,
        created_at="2021-01-01T00:00:00",
        status="pending",
    )


def test_task_serde(task):
    js = task.model_dump_json()

    new_task = TaskRead.from_json(js)

    assert new_task.user_id == "user"
    assert new_task.model_uri == "model"
    assert new_task.num_rounds == 10
    assert new_task.dataset_uris == ["uri1", "uri2"]
    assert new_task.status == Status.PENDING
    assert new_task.id == 1


def test_task_change_state(task):
    task_db = Task.model_validate(task)

    task_db.change_status("RUNNING")
    assert task_db.status == Status.RUNNING
    task_db.change_status("cancelled")
    assert task_db.status == Status.CANCELLED
