import os
from typing import Any


RABBIT_USERNAME = os.environ["RABBIT_USERNAME"]
RABBIT_PASSWORD = os.environ["RABBIT_PASSWORD"]
RABBIT_HOST = os.environ["RABBIT_HOST"]
RABBIT_PORT = os.environ["RABBIT_PORT"]

DB_MODE = os.getenv("DB_MODE", "sqlite")
DB_ECHO = bool(os.environ["DB_ECHO"])

POSTGRES_USERNAME = os.environ["POSTGRES_USERNAME"]
POSTGRES_PASSWORD = os.environ["POSTGRES_PASSWORD"]
POSTGRES_DB = os.environ["POSTGRES_DB"]
POSTGRES_HOST = os.environ["POSTGRES_HOST"]
POSTGRES_PORT = os.environ["POSTGRES_PORT"]

obj: dict[str, Any] = {}
