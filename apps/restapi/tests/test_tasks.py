import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from sqlmodel import Session, select, delete, SQLModel

from restapi.main import create_app
from restapi.db import engine, create_db_and_tables
from schemas.task import Task, TaskRead

app = create_app()
client = TestClient(app)


@pytest.fixture()
def session():
    create_db_and_tables()
    with Session(engine) as session:
        yield session
    SQLModel.metadata.drop_all(engine)


@pytest.fixture(autouse=True)
def clear_table(session):
    session.exec(delete(Task))
    session.commit()


@pytest.fixture(scope="session")
def task_attrs():
    return {
        "user_id": "1",
        "dataset_uris": ["uri1", "uri2"],
        "model_uri": "uri",
        "num_rounds": 10,
    }


def test_create_task(session, task_attrs):
    with patch(
        "restapi.routers.tasks.rabbitmq_client.publish_message"
    ) as mock_publish_message:
        response = client.post("/tasks/", json=task_attrs)

    assert response.status_code == 200
    for k in task_attrs.keys():
        assert response.json()[k] == task_attrs[k]

    tasks = session.exec(select(Task)).all()
    assert len(tasks) == 1
    mock_publish_message.assert_called_once_with(
        TaskRead.model_validate(tasks[0]).model_dump_json()
    )


def test_get_tasks_all(session, task_attrs):
    task = Task.model_validate(task_attrs)
    session.add(task)
    session.commit()
    session.refresh(task)

    task_view = task.model_copy()
    task_view.created_at = task_view.created_at.isoformat()

    response = client.get("/tasks/")
    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0] == task_view.model_dump()


def test_delete_tasks(session, task_attrs):
    task = Task.model_validate(task_attrs)
    session.add(task)
    session.commit()
    session.refresh(task)

    response = client.delete("/tasks/")
    assert response.status_code == 200
    tasks = session.exec(select(Task)).all()
    assert len(tasks) == 0


def test_cancel_task(session, task_attrs):
    task = Task.model_validate(task_attrs)
    session.add(task)
    session.commit()
    session.refresh(task)

    assert task.status == "pending"

    response = client.put(f"/tasks/cancel/{task.id}")
    assert response.status_code == 200
    session.refresh(task)

    assert task.status == "cancelled"
