import enum
from datetime import datetime, timezone
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
    selected_nodes: Optional[list[str]] = Field(default=None, sa_column=Column(JSON, nullable=True))
    validation_snapshot: Optional[dict] = Field(default=None, sa_column=Column(JSON, nullable=True))
    model_name: str = Field(nullable=False)
    model_version: int = Field(nullable=False)
    num_global_iterations: int = Field(nullable=False)
    run_name: str = Field(nullable=False)
    experiment_name: str = Field(nullable=False)
    SMPC_enabled: bool = Field(default=True)
    dp_mode: Optional[str] = Field(default=None)
    noise_multiplier: Optional[int] = Field(default=None)
    clipping_norm: Optional[int] = Field(default=None)
    dataset_meta_json: Optional[str] = Field(default=None, description="Serialised DCAT-AP metadata schema")
    node_dataset_map_json: Optional[str] = Field(
        default=None,
        description=(
            "Legacy dataset assignment retained for database compatibility. "
            "The current FL server resolves datasets directly from the catalogue."
        ),
    )
    n_samples_synthetic: Optional[int] = Field(
        default=10000,
        description="The number of synthetic patient records to generate post-training"
    )

class TaskRead(TaskBase, table=True):
    id: int = Field(default=None, primary_key=True, nullable=False)
    status: Status = Field(default=Status.PENDING, sa_column=Column(Enum(Status)))
    # Existing databases store UTC in TIMESTAMP WITHOUT TIME ZONE. asyncpg
    # requires a naive datetime for that column, even when the clock uses UTC.
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        nullable=False,
    )

    @model_validator(mode='after')
    def validate_security_protocol(self) -> Self:
        if self.dp_mode == "None":
            # SQLModel validates before SQLAlchemy instance state is attached.
            self.__dict__["dp_mode"] = None
        if self.dp_mode not in (None, "central_dp", "local_dp"):
            raise ValueError("Invalid dp_mode. Must be 'central_dp', 'local_dp', or None.")
        if self.dp_mode == "central_dp" and self.SMPC_enabled is True:
            raise ValueError("SMPC cannot be enabled with central_dp; model-local DP remains supported.")
        return self

    def change_status(self, new_status: str) -> None:
        self.status = Status[new_status.upper()]


class TaskCreate(TaskBase):
    @model_validator(mode="after")
    def validate_selected_nodes(self) -> Self:
        if self.selected_nodes is not None:
            if not self.selected_nodes or any(not name.strip() for name in self.selected_nodes):
                raise ValueError("selected_nodes must contain at least one nonblank node name")
            if len(set(self.selected_nodes)) != len(self.selected_nodes):
                raise ValueError("selected_nodes must not contain duplicates")
        return self
