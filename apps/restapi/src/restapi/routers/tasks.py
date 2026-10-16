import asyncio
import os
import requests
from typing import Sequence
from fastapi import APIRouter, Response, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from sqlmodel import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import ValidationError
from restapi.db import get_session
from restapi import config
from interfaces.rabbitmq_client import AsyncRabbitMQClient
from schemas.task import TaskCreate, TaskRead, TaskStatus

from restapi.auth import UserClaims, require_authentication

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
    current_user: UserClaims = Depends(require_authentication),
) -> Sequence[TaskRead]:
    result = await session.execute(select(TaskRead))
    return result.scalars().all()


@router.get("/{task_id}", response_model=TaskRead)
async def get_task_by_id(
    task_id: int, session: AsyncSession = Depends(get_session),
    current_user: UserClaims = Depends(require_authentication),
) -> JSONResponse | TaskRead:
    task = await session.get(TaskRead, task_id)
    if not task:
        return JSONResponse(status_code=404, content={"message": "Task not found"})
    return task


@router.get("/user/{user_id}", response_model=list[TaskRead])
async def get_tasks_by_user(
    user_id: str, session: AsyncSession = Depends(get_session),
    current_user: UserClaims = Depends(require_authentication),
) -> Sequence[TaskRead]:
    result = await session.execute(select(TaskRead).where(TaskRead.user_id == user_id))
    return result.scalars().all()


@router.post("/", response_model=TaskRead)
async def create_task(
    task: TaskCreate,
    session: AsyncSession = Depends(get_session),
    rabbitmqs: dict[str, AsyncRabbitMQClient] = Depends(get_rabbitmqs),
    current_user: UserClaims = Depends(require_authentication),
) -> TaskRead:
    try:
        # Ignore client-supplied snapshots: resolve and persist server-side at submission.
        task.validation_snapshot = None
        profile = os.getenv('VALIDATION_PROFILE', 'none')
        if profile in {'aml-full', 'scd-full'}:
            if task.use_case.lower() + '-full' != profile:
                raise HTTPException(422, 'Task disease does not match the deployed validation profile')
            from interfaces.catalogue_auth import catalogue_headers
            from interfaces.dataset_snapshot import select_snapshot
            def freeze():
                response = requests.get(config.DATA_CATALOGUE_URL.rstrip('/') + '/metadata',
                                        headers=catalogue_headers(), timeout=30)
                response.raise_for_status()
                snapshot = select_snapshot(response.json().get('datasets', []), task.use_case, task.selected_nodes)
                entry = next(iter(snapshot['nodes'].values()))
                published = requests.get(config.DATA_CATALOGUE_URL.rstrip('/') + '/evaluation-configurations/' + task.use_case,
                    params={'version': entry['configuration_version']}, headers=catalogue_headers(), timeout=30)
                published.raise_for_status()
                if published.json().get('documents') != entry['documents']:
                    raise ValueError('Dataset configuration does not match the published central version')
                privacy = entry['documents']['evaluation']['privacy_evaluation']
                if task.dp_mode == 'central_dp' and not privacy['differential_privacy']:
                    raise ValueError('Task DP mode does not match the frozen privacy configuration')
                return snapshot
            try:
                task.validation_snapshot = await asyncio.to_thread(freeze)
            except ValueError as error:
                raise HTTPException(422, str(error)) from error
            except (requests.RequestException, RuntimeError) as error:
                raise HTTPException(503, 'Unable to freeze dataset configuration from catalogue') from error
        db_task = TaskRead.model_validate(task)
        session.add(db_task)
        await session.commit()
        await session.refresh(db_task)

        # Publish message to RabbitMQ
        await rabbitmqs["dispatch"].publish_message(
            TaskRead.model_validate(db_task).model_dump_json()
        )

        return db_task
    except ValidationError as e:
        # Handle Pydantic validation errors
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Validation error: {str(e)}"
        )


@router.put("/cancel/{task_id}", response_model=TaskRead)
async def cancel_task(
    task_id: int,
    session: AsyncSession = Depends(get_session),
    rabbitmqs: dict[str, AsyncRabbitMQClient] = Depends(get_rabbitmqs),
    current_user: UserClaims = Depends(require_authentication),
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
    current_user: UserClaims = Depends(require_authentication),
) -> Response:
    task = await session.get(TaskRead, task_id)
    if not task:
        return JSONResponse(status_code=404, content={"message": "Task not found"})

    await session.delete(task)
    await session.commit()
    return JSONResponse({"message": f"Task {task_id} deleted"})


@router.delete("/")
async def delete_all_tasks(session: AsyncSession = Depends(get_session),
    current_user: UserClaims = Depends(require_authentication),
) -> Response:
    if not current_user.has_role("Admin"):
        raise HTTPException(status_code=403, detail="No permissions to access this endpoint")
    await session.execute(delete(TaskRead))
    await session.commit()
    return JSONResponse({"message": "All tasks deleted"})
