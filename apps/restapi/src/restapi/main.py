import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from restapi.routers import tasks, mlflow, keycloak_admin, dashboard
from restapi.services.lifespan import lifespan
from restapi.auth import keycloak


logger = structlog.get_logger()


def create_app() -> FastAPI:
    """Creates and configures the FastAPI app."""
    app = FastAPI(lifespan=lifespan)
    keycloak.add_swagger_config(app)
    app.include_router(tasks.router)
    app.include_router(dashboard.router)
    app.include_router(mlflow.router)
    app.include_router(mlflow.router_admin)
    app.include_router(keycloak_admin.router)


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
  # Adds OpenAPI support