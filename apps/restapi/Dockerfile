ARG CURRENT_VERSION
ARG BASE_IMAGE

FROM python:3.11-slim as builder

RUN pip install poetry

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN touch README.md

RUN poetry install --no-root && rm -rf $POETRY_CACHE_DIR

FROM ${BASE_IMAGE}:${CURRENT_VERSION} as runtime

RUN apt-get update && apt-get install -y curl

WORKDIR /app

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/common:/app/src"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}
COPY src/ ./src
COPY tests/ ./tests

CMD ["uvicorn", "src.restapi.main:app", "--host", "0.0.0.0", "--port", "8000"]
