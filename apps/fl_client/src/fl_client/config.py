import os

MLFLOW_URL: str = os.getenv("MLFLOW_URL")
NODE_NAME: str = os.getenv("NODE_NAME")
DATA_PATH: str = os.getenv("DATA_PATH", "/common/fl_models/iris/data.csv")
