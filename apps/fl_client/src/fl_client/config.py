import os

MLFLOW_URL: str = os.environ["MLFLOW_URL"]
NODE_NAME: str = os.environ["NODE_NAME"]
# A deployment represents one FL network and therefore one use case (AML or
# SCD).  Keep this optional for backwards compatibility with older manifests;
# an unset value means that the node accepts the network's use case.
FL_USE_CASE: str = os.getenv("FL_USE_CASE", "").strip()
DATA_PATH: str = os.getenv("DATA_PATH", "/common/fl_models/iris/data.csv")
DATA_PATH_AML: str = os.getenv("DATA_PATH_AML", "/common/fl_models/AML_public/AML_preprocessed.csv")
#FL_USE_CASES =  ["AML", "SCD"]
#FL_NODES =  os.getenv("FL_NODES") #IDS []?
MINIO_STORAGE_URL =  os.getenv("MINIO_STORAGE_URL")
