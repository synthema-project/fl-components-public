from pydantic import BaseModel


class Task(BaseModel):
    id: int
    use_case: str
    model_name: str
    model_version: int
    experiment_name: str
    run_name: str
