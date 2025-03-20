import pytest
from unittest.mock import AsyncMock
from httpx import AsyncClient  # Use AsyncClient for async FastAPI testing
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select, delete

from restapi.main import create_app
from restapi.db import get_session, create_db_and_tables, drop_db_and_tables
from restapi.routers.tasks import config
from interfaces.rabbitmq_client import AsyncRabbitMQClient
from schemas.task import TaskRead, Status

app = create_app()


@pytest_asyncio.fixture(scope="session")
def task_attrs():
    """Provides a valid task attributes dictionary."""
    return {
        "user_id": "1",
        "use_case": "use_case",
        "model_name": "model_name",
        "model_version": 1,
        "num_global_iterations": 10,
        "run_name": "run_name",
        "experiment_name": "experiment_name",
    }


@pytest_asyncio.fixture
async def async_session():
    """Provides an async session fixture."""
    await create_db_and_tables()
    async for session in get_session():
        yield session
    await drop_db_and_tables()


@pytest_asyncio.fixture
def rabbitmq():
    """Mocks the RabbitMQ client."""
    mock_rabbitmq_dispatch = AsyncMock(AsyncRabbitMQClient)
    mock_rabbitmq_status = AsyncMock(AsyncRabbitMQClient)
    config.obj["rabbitmq_dispatch"] = mock_rabbitmq_dispatch
    config.obj["rabbitmq_status"] = mock_rabbitmq_status
    return mock_rabbitmq_dispatch, mock_rabbitmq_status


@pytest_asyncio.fixture(autouse=True)
async def clear_table(async_session: AsyncSession):
    """Clears the task table before each test."""
    async for session in get_session():
        await session.execute(delete(TaskRead))
        await session.commit()


@pytest_asyncio.fixture
async def async_client():
    """Provides an async test client for FastAPI."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.mark.asyncio
async def test_create_task(async_client, async_session, task_attrs, rabbitmq):
    """Tests creating a new task and publishing a structured message."""
    response = await async_client.post("/tasks/", json=task_attrs)
    assert response.status_code == 200

    # Check the response
    for k in task_attrs.keys():
        assert response.json()[k] == task_attrs[k]

    # Check the database
    async for session in get_session():
        tasks = (await session.execute(select(TaskRead))).scalars().all()
        assert len(tasks) == 1

    # Validate RabbitMQ message
    rabbitmq[0].publish_message.assert_called_once()


@pytest.mark.asyncio
async def test_get_all_tasks(async_client, async_session, task_attrs):
    """Tests retrieving all tasks."""
    async for session in get_session():
        task = TaskRead(**task_attrs)
        session.add(task)
        await session.commit()
        await session.refresh(task)

    response = await async_client.get("/tasks/")
    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["user_id"] == task_attrs["user_id"]


@pytest.mark.asyncio
async def test_delete_one_task(async_client, async_session, task_attrs):
    """Tests deleting a single task."""
    async for session in get_session():
        task = TaskRead(**task_attrs)
        session.add(task)
        await session.commit()
        await session.refresh(task)

        response = await async_client.delete(f"/tasks/{task.id}")
        assert response.status_code == 200

        tasks = (await session.execute(select(TaskRead))).scalars().all()
        assert len(tasks) == 0


@pytest.mark.asyncio
async def test_delete_all_tasks(async_client, async_session, task_attrs):
    """Tests deleting all tasks."""
    async for session in get_session():
        task = TaskRead(**task_attrs)
        session.add(task)
        await session.commit()
        await session.refresh(task)

    response = await async_client.delete("/tasks/")
    assert response.status_code == 200

    async for session in get_session():
        tasks = (await session.execute(select(TaskRead))).scalars().all()
        assert len(tasks) == 0


@pytest.mark.asyncio
async def test_cancel_task(async_client, async_session, rabbitmq, task_attrs):
    """Tests canceling a task and checking status update."""
    async for session in get_session():
        task = TaskRead(**task_attrs)
        session.add(task)
        await session.commit()
        await session.refresh(task)

        assert task.status == Status.PENDING.value

        response = await async_client.put(f"/tasks/cancel/{task.id}")
        assert response.status_code == 200

        assert rabbitmq[1].publish_message.called_once()
