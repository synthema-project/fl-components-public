from contextlib import asynccontextmanager
from typing import Any
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


from restapi.db import create_db_and_tables
from restapi.routers import tasks
from restapi import config

from interfaces import rabbitmq_client


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> Any:
        rabbitmq_client.configure(
            config.RABBIT_USERNAME,
            config.RABBIT_PASSWORD,
            host=config.RABBIT_HOST,
            port=int(config.RABBIT_PORT),
        )
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
