import enum
from datetime import datetime
from typing import Optional, Self
from sqlmodel import SQLModel, Field, Column, Enum
from pydantic import BaseModel, model_validator


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
    SMPC_enabled: bool = Field(default=False)
    dp_mode: Optional[str] = Field(default=None)
    noise_multiplier: Optional[int] = Field(default=None)
    clipping_norm: Optional[int] = Field(default=None)
    dataset_meta_json: Optional[str] = Field(default=None, description="Serialised DCAT-AP metadata schema")
    node_dataset_map_json: Optional[str] = Field(
        default=None,
        description=(
            "JSON-encoded {node_name: [minio_url, ...]} dataset assignment for this "
            "task's use_case, resolved from the data catalogue at task-creation time. "
            "Broadcast to clients so each participates/loads by its NODE_NAME."
        ),
    )
    n_samples_synthetic: Optional[int] = Field(
        default=10000,
        description="The number of synthetic patient records to generate post-training"
    )

class TaskRead(TaskBase, table=True):
    id: int = Field(default=None, primary_key=True, nullable=False)
    status: Status = Field(default=Status.PENDING, sa_column=Column(Enum(Status)))
    created_at: datetime = Field(default_factory=datetime.now, nullable=False)

    @model_validator(mode='after')
    def validate_security_protocol(self) -> Self:
        if self.dp_mode == "None":
            self.dp_mode = None
        if self.dp_mode not in (None, "central_dp", "local_dp"):
            raise ValueError("Invalid dp_mode. Must be 'central_dp', 'local_dp', or None.")
        if self.dp_mode in ("central_dp", "local_dp") and self.SMPC_enabled is True:
            raise ValueError("SMPC cannot be enabled when dp_mode is set.")
        return self

    def change_status(self, new_status: str) -> None:
        self.status = Status[new_status.upper()]


class TaskCreate(TaskBase):
    pass
