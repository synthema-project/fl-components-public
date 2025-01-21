import os

MLFLOW_URL: str = os.environ["MLFLOW_URL"]
NODE_NAME: str = os.environ["NODE_NAME"]
DATA_PATH: str = os.getenv("DATA_PATH", "/common/fl_models/iris/data.csv")
