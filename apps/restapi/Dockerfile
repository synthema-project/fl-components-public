# syntax=docker/dockerfile:1.10
ARG CURRENT_VERSION=latest
ARG BASE_IMAGE=fl-common

FROM python:3.11-slim as builder

COPY --from=ghcr.io/astral-sh/uv:0.11.19 /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY common/pyproject.toml common/pyproject.toml
COPY apps/fl_client/pyproject.toml apps/fl_client/pyproject.toml
COPY apps/fl_server/pyproject.toml apps/fl_server/pyproject.toml
COPY apps/restapi/pyproject.toml apps/restapi/pyproject.toml
COPY apps/status_listener/pyproject.toml apps/status_listener/pyproject.toml
COPY e2e/pyproject.toml e2e/pyproject.toml

RUN uv venv .venv
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

ARG SYNTHEMA_PYPI_URL=https://pypi.synthema.rid-intrasoft.eu/simple/
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=secret,id=synthema_username,env=UV_INDEX_SYNTHEMA_USERNAME \
    --mount=type=secret,id=synthema_password,env=UV_INDEX_SYNTHEMA_PASSWORD \
    set -- --index "synthema=$SYNTHEMA_PYPI_URL"; \
    uv sync --locked --package synthema-fl-restapi --no-install-workspace "$@" --group dev

FROM ${BASE_IMAGE}:${CURRENT_VERSION} as runtime

RUN apt-get update && apt-get install -y curl

WORKDIR /app

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/common:/app/src"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}
COPY apps/restapi/src/ ./src
COPY apps/restapi/tests/ ./tests

# Fail the image build if the deployed ASGI entry point cannot load with auth enabled.
RUN python tests/test_entrypoint.py

CMD ["uvicorn", "restapi.main:app", "--host", "0.0.0.0", "--port", "8000"]
