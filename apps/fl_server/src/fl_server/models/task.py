from pydantic import BaseModel


class Task(BaseModel):
    id: int
    use_case: str
    model_name: str
    model_version: int
    experiment_name: str
    run_name: str
    num_global_iterations: int


def _get_dummy_task() -> Task:
    return Task(
        id=1,
        use_case="iris",
        model_name="iris_model",
        model_version=3,
        experiment_name="iris",
        run_name="run1",
        num_global_iterations=3,
    )
