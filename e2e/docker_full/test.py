import os
import time

import mlflow.runs

os.environ["MLFLOW_URL"] = "http://mlflow"

import mlflow
import requests

from fl_models.iris.fl_model import log_model

log_model()

requests.post(
    "http://synthema-fl-restapi:8000/tasks",
    json={
        "user_id": "1",
        "use_case": "iris",
        "model_name": "iris_model",
        "model_version": 1,
        "num_global_iterations": 3,
        "run_name": "test_run",
        "experiment_name": "Default",
    },
)

# Check that the model was logged
trials = 0
run_id = None
runs = []
while trials < 10 and len(runs) == 0:
    runs = mlflow.search_runs(
        filter_string='attributes.run_name="test_run"', output_format="list"
    )
    try:
        run_id = runs[0].info.run_id
    except IndexError:
        print("Run not found. Retrying...")
        trials += 1
        time.sleep(10)

if run_id is not None:
    print(f"Run found: {run_id}")
else:
    print("No runs found.")
    exit(1)
run = mlflow.get_run(run_id)
while run.info.status in {"RUNNING", "SCHEDULED"}:
    time.sleep(10)
    print(f"Run status: {run.info.status}")
    run = mlflow.get_run(run_id)

if run.info.status != "FINISHED":
    print(f"Expected run status to be FINISHED, but got {run.info.status}")
    exit(1)
