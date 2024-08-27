from typing import Sequence
from fastapi import APIRouter, Response
from fastapi.responses import JSONResponse
from sqlmodel import Session, select, delete

from restapi.db import engine

from schemas.task import TaskCreate, TaskRead, Task
from interfaces import rabbitmq_client

router = APIRouter(prefix="/tasks", tags=["tasks"], dependencies=None)


@router.get("/", response_model=list[TaskRead])
async def get_tasks_all() -> Sequence[Task]:
    with Session(engine) as session:
        tasks = session.exec(select(Task)).all()
        return tasks


@router.get("/{task_id}", response_model=TaskRead)
async def get_task_by_id(task_id: int) -> JSONResponse | Task:
    with Session(engine) as session:
        task = session.get(Task, task_id)
        if not task:
            return JSONResponse(status_code=404, content="Task not found")
        return task


@router.get("/user/{user_id}", response_model=list[TaskRead])
async def get_tasks_by_user(user_id: str) -> Sequence[Task]:
    with Session(engine) as session:
        tasks = session.exec(select(Task).where(Task.user_id == user_id)).all()
        return tasks


@router.post("/", response_model=TaskRead)
async def create_task(task: TaskCreate) -> Task:
    with Session(engine) as session:
        db_task = Task.model_validate(task)
        session.add(db_task)
        session.commit()
        session.refresh(db_task)
        rabbitmq_client.publish_message(
            TaskRead.model_validate(db_task).model_dump_json()
        )
        return db_task


@router.put("/cancel/{task_id}", response_model=TaskRead)
async def cancel_task(task_id: int) -> Task | Response:
    with Session(engine) as session:
        task = session.get(Task, task_id)
        if not task:
            return JSONResponse(status_code=404, content="Task not found")
        task.change_status("cancelled")
        session.commit()
        session.refresh(task)
        return task


@router.delete("/")
async def delete_all_tasks() -> Response:
    with Session(engine) as session:
        session.exec(delete(Task))  # type: ignore[call-overload]
        session.commit()
        return JSONResponse(status_code=200, content="All tasks deleted")
