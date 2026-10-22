import enum
from datetime import datetime
from typing import Optional, Self

from sqlmodel import SQLModel, Field, Column, Enum
from pydantic import BaseModel, model_validator
from sqlalchemy import JSON


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
    dataset_meta_json: Optional[str] = None
    selected_nodes: Optional[list[str]] = Field(default=None, sa_column=Column(JSON, nullable=True))
    validation_snapshot: Optional[dict] = None
    model_name: str = Field(nullable=False)
    model_version: int = Field(nullable=False)
    num_global_iterations: int = Field(nullable=False)
    run_name: str = Field(nullable=False)
    experiment_name: str = Field(nullable=False)
    SMPC_enabled: bool = Field(default=False)
    dp_mode: Optional[str] = Field(default=None)
    noise_multiplier: Optional[int] = Field(default=None)
    clipping_norm: Optional[int] = Field(default=None)
    num_synthetic_samples: int = Field(default=0, nullable=False)


class TaskRead(TaskBase):

    id: int = Field(default=None, primary_key=True, nullable=False)
    status: Status = Field(default=Status.PENDING, sa_column=Column(Enum(Status)))
    created_at: datetime = Field(default_factory=datetime.now, nullable=False)

    @model_validator(mode="before")
    @classmethod
    def read_synthetic_sample_count(cls, values):
        if isinstance(values, dict):
            values = dict(values)
            canonical = values.get("n_samples_synthetic")
            legacy = values.get("num_synthetic_samples")
            if canonical is not None and legacy is not None and canonical != legacy:
                raise ValueError("Conflicting synthetic sample counts")
            count = canonical if canonical is not None else legacy
            if count is None:
                count = 10000
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("Synthetic sample count must be a non-negative integer")
            values["num_synthetic_samples"] = count
        return values

    @model_validator(mode="after")
    def validate_security_protocol(self) -> Self:
        if self.dp_mode == "None":
            self.dp_mode = None
        if self.dp_mode not in (None, "central_dp", "local_dp"):
            raise ValueError("Invalid dp_mode. Must be 'central_dp', 'local_dp', or None.")
        if self.dp_mode == "central_dp" and self.SMPC_enabled:
            raise ValueError("SMPC cannot be enabled with central_dp; model-local DP remains supported.")
        return self

    def change_status(self, new_status: str) -> None:
        self.status = Status[new_status.upper()]


class TaskCreate(TaskBase):
    pass
