"""
Passthrough schema module to eradicate Schema Drift (DRY violation).
Mathematically delegates all Pydantic and SQLModel definitions to the
immutable DevSecOps SDK in the common repository.
"""

# Strict import from the common library (Single Source of Truth)
# This guarantees the FL Server loads the exact same fields (dataset_meta_json)
# as the REST API and the RabbitMQ transit layer.
from schemas.task import (
    Status,
    TaskStatus,
    TaskBase,
    TaskRead,
    TaskCreate,
)

# Explicitly re-export the definitions so downstream fl_server imports do not break
__all__ = ["Status", "TaskStatus", "TaskBase", "TaskRead", "TaskCreate"]
