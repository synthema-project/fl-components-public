import json
from typing import Optional, Sequence
from fastapi import APIRouter, Response, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials
from sqlmodel import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import ValidationError
import requests
from restapi.db import get_session
from restapi import config
from restapi.helpers.data_catalogue import fetch_use_case_datasets
from interfaces.rabbitmq_client import AsyncRabbitMQClient
from schemas.task import TaskCreate, TaskRead, TaskStatus

from restapi.auth import UserClaims, oauth2_scheme, require_authentication

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
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(oauth2_scheme),
) -> TaskRead:
    try:
        # Resolve the per-node dataset assignment for this use case from the data
        # catalogue, forwarding the caller's token (the fl-server never queries the
        # catalogue itself — it receives this map inside the task message).
        bearer_token = credentials.credentials if credentials else None
        try:
            node_dataset_map = fetch_use_case_datasets(task.use_case, bearer_token)
        except requests.RequestException as e:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Could not reach the data catalogue: {e}",
            )
        if not node_dataset_map:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"No datasets are registered for use case '{task.use_case}'. "
                    f"Register datasets (per node) in the data catalogue first."
                ),
            )

        # Restrict the run to the nodes the caller selected (if any). Absent /
        # empty selection keeps the previous behaviour of using every node.
        selected_nodes = getattr(task, "selected_nodes", None)
        if selected_nodes:
            unknown = [n for n in selected_nodes if n not in node_dataset_map]
            if unknown:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"Selected node(s) {unknown} are not registered for use "
                        f"case '{task.use_case}'. Available: {list(node_dataset_map)}."
                    ),
                )
            node_dataset_map = {n: node_dataset_map[n] for n in selected_nodes}

        db_task = TaskRead.model_validate(task)
        db_task.node_dataset_map_json = json.dumps(node_dataset_map)
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
