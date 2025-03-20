import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from restapi.routers import tasks
from restapi.services.lifespan import lifespan

logger = structlog.get_logger()


def create_app() -> FastAPI:
    """Creates and configures the FastAPI app."""
    app = FastAPI(lifespan=lifespan)

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


# FastAPI app entrypoint for Uvicorn
app = create_app()
