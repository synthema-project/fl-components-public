from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlmodel import SQLModel
from schemas.migrations import upgrade_task_node_selection
from status_listener import config
from typing import AsyncGenerator

url = f"postgresql+asyncpg://{config.POSTGRES_USERNAME}:{config.POSTGRES_PASSWORD}@{config.POSTGRES_HOST}:{config.POSTGRES_PORT}/{config.POSTGRES_DB}"

engine = create_async_engine(url, echo=config.DB_ECHO)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to provide an async session."""
    async with SessionLocal() as session:
        yield session


async def create_db_and_tables() -> None:
    """Creates database tables asynchronously."""
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
        await conn.run_sync(upgrade_task_node_selection)


async def drop_db_and_tables() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)
