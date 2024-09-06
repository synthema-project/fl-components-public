from sqlmodel import create_engine, SQLModel

from restapi import config

if config.DB_MODE == "sqlite":
    url = "sqlite:///./test.db"
    connect_args = {"check_same_thread": False}  # for sqlite

elif config.DB_MODE == "postgresql":
    url = f"postgresql://{config.POSTGRES_USERNAME}:{config.POSTGRES_PASSWORD}@{config.POSTGRES_HOST}:{config.POSTGRES_PORT}/{config.POSTGRES_DB}"
    connect_args = {}

else:
    raise ValueError(f"Unsupported DB_MODE: {config.DB_MODE}")

engine = create_engine(url, echo=config.DB_ECHO, connect_args=connect_args)


def create_db_and_tables() -> None:
    SQLModel.metadata.create_all(engine)
