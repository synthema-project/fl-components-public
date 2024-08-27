from typing import Any
from sqlmodel import create_engine, SQLModel

from restapi import config

postgres_arg = f"{config.POSTGRES_USERNAME}:{config.POSTGRES_PASSWORD}@{config.POSTGRES_HOST}:{config.POSTGRES_PORT}/{config.POSTGRES_DB}"
postgres_url = f"postgresql://{postgres_arg}"

# connect_args = {"check_same_thread": False} # for sqlite
connect_args: dict[Any, Any] = {}  # for postgres
engine = create_engine(
    postgres_url, echo=config.POSTGRES_ECHO, connect_args=connect_args
)


def create_db_and_tables() -> None:
    SQLModel.metadata.create_all(engine)
