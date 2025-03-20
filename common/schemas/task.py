import enum
from datetime import datetime

from sqlmodel import SQLModel, Field, Column, Enum
from pydantic import BaseModel


class Status(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStatus(BaseModel):
    id: int
    status: Status


class TaskBase(SQLModel):
    user_id: str = Field(nullable=False)
    # dataset_uris: List[str] = Field(default=None, sa_column=Column(ARRAY(String())))
    use_case: str = Field(nullable=False)
    model_name: str = Field(nullable=False)
    model_version: int = Field(nullable=False)
    num_global_iterations: int = Field(nullable=False)
    run_name: str = Field(nullable=False)
    experiment_name: str = Field(nullable=False)


class TaskRead(TaskBase, table=True):
    id: int = Field(default=None, primary_key=True, nullable=False)
    status: Status = Field(default=Status.PENDING, sa_column=Column(Enum(Status)))
    created_at: datetime = Field(default_factory=datetime.now, nullable=False)

    def change_status(self, new_status: str) -> None:
        self.status = Status[new_status.upper()]


class TaskCreate(TaskBase):
    pass
