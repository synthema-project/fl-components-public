import os
from typing import Any


def get_env_variable(name: str) -> str:
    """Fetches an environment variable and raises an exception if it's missing."""
    value = os.getenv(name)
    if value is None:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


# Ensure mandatory RabbitMQ configuration is set
RABBIT_USERNAME = get_env_variable("RABBIT_USERNAME")
RABBIT_PASSWORD = get_env_variable("RABBIT_PASSWORD")
RABBIT_HOST = get_env_variable("RABBIT_HOST")
RABBIT_PORT = int(get_env_variable("RABBIT_PORT"))
RABBIT_QUEUE = get_env_variable("RABBIT_QUEUE")

DB_ECHO = bool(os.getenv("DB_ECHO", "true"))
# Ensure mandatory Postgres configuration is set
POSTGRES_USERNAME = get_env_variable("POSTGRES_USERNAME")
POSTGRES_PASSWORD = get_env_variable("POSTGRES_PASSWORD")
POSTGRES_DB = get_env_variable("POSTGRES_DB")
POSTGRES_HOST = get_env_variable("POSTGRES_HOST")
POSTGRES_PORT = int(get_env_variable("POSTGRES_PORT"))

# Global object for shared runtime config
obj: dict[str, Any] = {}
