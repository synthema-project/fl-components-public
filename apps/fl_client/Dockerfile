ARG CURRENT_VERSION=latest
ARG BASE_IMAGE=fl-common

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

WORKDIR /app

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/common:/app/src"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}
# Install grpcio-tools and protobuf in the runtime container
RUN pip install grpcio-tools protobuf
COPY src/ ./src
COPY tests/ ./tests

# Compile the protobuf file in the appropriate directory
RUN python -m grpc_tools.protoc -I/app/src/fl_client --python_out=/app/src/fl_client --grpc_python_out=/app/src/fl_client /app/src/fl_client/smpc.proto
RUN sed -i 's/^import smpc_pb2 as smpc__pb2$/import fl_client.smpc_pb2 as smpc__pb2/' /app/src/fl_client/smpc_pb2_grpc.py

CMD ["flower-client-app", "--insecure", "src.fl_client.main:app"]