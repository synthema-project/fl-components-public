import enum
from datetime import datetime
from typing import Optional

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
    """Base schema for FL tasks, shared between read and create variants."""

    user_id: str = Field(nullable=False)
    # dataset_uris: List[str] = Field(default=None, sa_column=Column(ARRAY(String())))
    use_case: str = Field(nullable=False)
    model_name: str = Field(nullable=False)
    model_version: int = Field(nullable=False)
    num_global_iterations: int = Field(nullable=False)
    run_name: str = Field(nullable=False)
    experiment_name: str = Field(nullable=False)
    dataset_meta_json: Optional[str] = Field(
        default=None, description="Serialised DCAT-AP metadata schema"
    )
    n_samples_synthetic: Optional[int] = Field(
        default=10000,
        description="The number of synthetic patient records to generate post-training"
    )

    # --- Privacy / security configuration ---
    SMPC_enabled: bool = Field(
        default=False, description="Enable Secure Multi-Party Computation"
    )
    dp_mode: Optional[str] = Field(
        default=None, description="Differential-privacy mode (e.g. 'central_dp')"
    )
    noise_multiplier: Optional[float] = Field(
        default=None, description="DP noise multiplier"
    )
    clipping_norm: Optional[float] = Field(
        default=None, description="DP gradient-clipping norm"
    )


class TaskRead(TaskBase, table=True):
    id: int = Field(default=None, primary_key=True, nullable=False)
    status: Status = Field(default=Status.PENDING, sa_column=Column(Enum(Status)))
    created_at: datetime = Field(default_factory=datetime.now, nullable=False)

    def change_status(self, new_status: str) -> None:
        self.status = Status[new_status.upper()]


class TaskCreate(TaskBase):
    pass
