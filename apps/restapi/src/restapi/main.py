from contextlib import asynccontextmanager
from threading import Event
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from restapi import config
from restapi.status_updates import StatusUpdates
from restapi.db import create_db_and_tables
from restapi.routers import tasks

from interfaces.rabbitmq_client import setup_rabbitmq


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator:
        # init rabbitmq
        config.obj["rabbitmq_dispatch"] = setup_rabbitmq(
            config.RABBIT_USERNAME,
            config.RABBIT_PASSWORD,
            config.RABBIT_HOST,
            int(config.RABBIT_PORT),
            1000,
            "dispatch",
            True,
        )
        config.obj["rabbitmq_status"] = setup_rabbitmq(
            config.RABBIT_USERNAME,
            config.RABBIT_PASSWORD,
            config.RABBIT_HOST,
            int(config.RABBIT_PORT),
            1000,
            "status",
            True,
        )
        # init status updates thread
        event = Event()
        thread = StatusUpdates(event)
        thread.start()
        # init db
        create_db_and_tables()
        yield

    app = FastAPI(
        swagger_ui_init_oauth={"usePkceWithAuthorizationCodeGrant": True},
        lifespan=lifespan,
    )
    # app.include_router(queue.router)
    app.include_router(tasks.router)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def root() -> dict:
        return {"message": "Pong"}

    return app


def startup_app(app: FastAPI, host: str, port: int) -> None:
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Start the FastAPI app")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to run the FastAPI app"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the FastAPI app"
    )
    args = parser.parse_args()

    app = create_app()
    startup_app(app, args.host, args.port)
