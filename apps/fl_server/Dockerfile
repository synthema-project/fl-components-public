ARG CURRENT_VERSION=latest
ARG BASE_IMAGE=fl-common

FROM python:3.11-slim as builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

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

RUN --mount=type=cache,target=/root/.cache/uv \
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