ARG CURRENT_VERSION=latest
ARG BASE_IMAGE=fl-common

FROM python:3.11-slim as builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml ./
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
# Install grpcio-tools and protobuf in the runtime container
RUN pip install grpcio-tools protobuf
COPY src/ ./src
COPY tests/ ./tests

# Compile the protobuf file in the appropriate directory
RUN python -m grpc_tools.protoc -I/app/src/fl_client --python_out=/app/src/fl_client --grpc_python_out=/app/src/fl_client /app/src/fl_client/smpc.proto
RUN sed -i 's/^import smpc_pb2 as smpc__pb2$/import fl_client.smpc_pb2 as smpc__pb2/' /app/src/fl_client/smpc_pb2_grpc.py

CMD ["flower-client-app", "--insecure", "src.fl_client.main:app"]