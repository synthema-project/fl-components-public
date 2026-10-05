# syntax=docker/dockerfile:1.10
ARG CURRENT_VERSION=latest
ARG BASE_IMAGE=fl-common

FROM python:3.11-slim as builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends git libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
# Vendored model-store-interface wheel (built & copied here by scripts/images.py).
# The package lives in a sibling repo unreachable from this build context, so it is
# resolved from a local find-links directory instead of an index.
COPY wheels/ /wheels/
RUN touch README.md

RUN uv venv .venv
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV UV_FIND_LINKS=/wheels

# Resolve MSI from the exact validated wheel, including a version newer than
# the compatibility baseline declared in pyproject.toml.
RUN python -c "from pathlib import Path; import json; w=list(Path('/wheels').glob('model_store_interface-*.whl')); assert len(w)==1; p=Path('pyproject.toml'); p.write_text(p.read_text()+'\n[tool.uv]\noverride-dependencies = '+json.dumps(['model-store-interface @ '+w[0].as_uri()])+'\n')"
ARG VALIDATION_PYPI_URL=https://pypi.synthema.rid-intrasoft.eu/simple/
ARG PRIVACY_PYPI_URL=https://pypi.synthema.rid-intrasoft.eu/simple/
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=secret,id=synthema_username,env=UV_INDEX_SYNTHEMA_USERNAME \
    --mount=type=secret,id=synthema_password,env=UV_INDEX_SYNTHEMA_PASSWORD \
    --mount=type=secret,id=synthema_username,env=UV_INDEX_VALIDATION_USERNAME \
    --mount=type=secret,id=synthema_password,env=UV_INDEX_VALIDATION_PASSWORD \
    --mount=type=secret,id=synthema_username,env=UV_INDEX_PRIVACY_USERNAME \
    --mount=type=secret,id=synthema_password,env=UV_INDEX_PRIVACY_PASSWORD \
    export UV_INDEX_VALIDATION_URL="$VALIDATION_PYPI_URL" UV_INDEX_PRIVACY_URL="$PRIVACY_PYPI_URL"; \
    uv sync --no-install-project --group dev

FROM ${BASE_IMAGE}:${CURRENT_VERSION} as runtime

WORKDIR /app

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/common:/app/src"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}
COPY src/ ./src
COPY tests/ ./tests

CMD ["python", "-m", "src.fl_server.main"]