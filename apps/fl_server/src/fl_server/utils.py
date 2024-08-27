from interfaces.mlflow_client import create_child_run, create_parent_run


def create_mlflow_runs(run_name: str, experiment_name: str) -> tuple[str, str, str]:
    parent_run_id, experiment_id = create_parent_run(run_name, experiment_name)
    child_run_id = create_child_run(experiment_id, parent_run_id, "global")
    return experiment_id, parent_run_id, child_run_id
