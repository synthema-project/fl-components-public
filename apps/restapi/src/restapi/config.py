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
RABBIT_PORT = int(get_env_variable("RABBIT_PORT"))  # Ensure integer conversion

DB_ECHO = bool(os.getenv("DB_ECHO", "true"))
DB_MODE = get_env_variable("DB_MODE")
# Ensure mandatory Postgres configuration is set
if DB_MODE == "postgresql":
    POSTGRES_USERNAME = get_env_variable("POSTGRES_USERNAME")
    POSTGRES_PASSWORD = get_env_variable("POSTGRES_PASSWORD")
    POSTGRES_DB = get_env_variable("POSTGRES_DB")
    POSTGRES_HOST = get_env_variable("POSTGRES_HOST")
    POSTGRES_PORT = int(get_env_variable("POSTGRES_PORT"))


KEYCLOAK_SERVER_URL = get_env_variable("KEYCLOAK_SERVER_URL")
KEYCLOAK_CLIENT_ID = get_env_variable("KEYCLOAK_CLIENT_ID")
KEYCLOAK_CLIENT_SECRET = get_env_variable("KEYCLOAK_CLIENT_SECRET")
KEYCLOAK_REALM_NAME = get_env_variable("KEYCLOAK_REALM_NAME")
KEYCLOAK_REDIRECT_URI = get_env_variable("KEYCLOAK_REDIRECT_URI")
KEYCLOAK_AUTHORIZED_ROLE = get_env_variable("KEYCLOAK_AUTHORIZED_ROLE")
KEYCLOAK_AUTHORIZED_GROUP = get_env_variable("KEYCLOAK_AUTHORIZED_GROUP") # Keycloak uses `/` prefix for group paths
KEYCLOAK_ADMIN_CLIENT_SECRET = get_env_variable("KEYCLOAK_ADMIN_CLIENT_SECRET")

# Global object for shared runtime config
obj: dict[str, Any] = {}
