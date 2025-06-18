from typing import Sequence
from fastapi import APIRouter, Response, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlmodel import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from restapi.db import get_session
from restapi import config
from interfaces.rabbitmq_client import AsyncRabbitMQClient
from schemas.task import TaskCreate, TaskRead, TaskStatus

#from restapi.auth import verify_token, verify_token_and_role
from restapi.auth import get_current_user, get_current_user_with_restricted_role

router = APIRouter(prefix="/tasks", tags=["tasks"])


async def get_rabbitmqs() -> dict[str, AsyncRabbitMQClient]:
    """Dependency to fetch RabbitMQ dispatch client"""
    rabbitmq_dispatch = config.obj.get("rabbitmq_dispatch")
    if not isinstance(rabbitmq_dispatch, AsyncRabbitMQClient):
        raise RuntimeError("RabbitMQ dispatch client not found")
    rabbitmq_status = config.obj.get("rabbitmq_dispatch")
    if not isinstance(rabbitmq_status, AsyncRabbitMQClient):
        raise RuntimeError("RabbitMQ dispatch client not found")
    return {"status": rabbitmq_status, "dispatch": rabbitmq_dispatch}


@router.get("/", response_model=list[TaskRead])
async def get_tasks_all(
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> Sequence[TaskRead]:
    result = await session.execute(select(TaskRead))
    return result.scalars().all()


@router.get("/{task_id}", response_model=TaskRead)
async def get_task_by_id(
    task_id: int, session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> JSONResponse | TaskRead:
    task = await session.get(TaskRead, task_id)
    if not task:
        return JSONResponse(status_code=404, content={"message": "Task not found"})
    return task


@router.get("/user/{user_id}", response_model=list[TaskRead])
async def get_tasks_by_user(
    user_id: str, session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> Sequence[TaskRead]:
    result = await session.execute(select(TaskRead).where(TaskRead.user_id == user_id))
    return result.scalars().all()


@router.post("/", response_model=TaskRead)
async def create_task(
    task: TaskCreate,
    session: AsyncSession = Depends(get_session),
    rabbitmqs: dict[str, AsyncRabbitMQClient] = Depends(get_rabbitmqs),
    user: dict = Depends(get_current_user_with_restricted_role),
) -> TaskRead:
    db_task = TaskRead.model_validate(task)
    session.add(db_task)
    await session.commit()
    await session.refresh(db_task)

    # Publish message to RabbitMQ
    await rabbitmqs["dispatch"].publish_message(
        TaskRead.model_validate(db_task).model_dump_json()
    )

    return db_task


@router.put("/cancel/{task_id}", response_model=TaskRead)
async def cancel_task(
    task_id: int,
    session: AsyncSession = Depends(get_session),
    rabbitmqs: dict[str, AsyncRabbitMQClient] = Depends(get_rabbitmqs),
    user: dict = Depends(get_current_user),
) -> TaskRead | Response:

    task = await session.get(TaskRead, task_id)
    if not task:
        return JSONResponse(status_code=404, content={"message": "Task not found"})

    task_status = TaskStatus(id=task.id, status=task.status)
    await rabbitmqs["status"].publish_message(task_status.model_dump_json())
    await session.commit()
    await session.refresh(task)
    return task


@router.delete("/{task_id}")
async def delete_task(
    task_id: int, session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> Response:
    task = await session.get(TaskRead, task_id)
    if not task:
        return JSONResponse(status_code=404, content={"message": "Task not found"})

    await session.delete(task)
    await session.commit()
    return JSONResponse({"message": f"Task {task_id} deleted"})


@router.delete("/")
async def delete_all_tasks(session: AsyncSession = Depends(get_session),
    user = Depends(get_current_user_with_restricted_role(required_roles=["central_node_admin"])),
) -> Response:
    await session.execute(delete(TaskRead))
    await session.commit()
    return JSONResponse({"message": "All tasks deleted"})
