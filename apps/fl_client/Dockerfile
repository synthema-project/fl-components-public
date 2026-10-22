# syntax=docker/dockerfile:1.10
ARG CURRENT_VERSION=latest
ARG BASE_IMAGE=fl-common

FROM python:3.11-slim as builder

COPY --from=ghcr.io/astral-sh/uv:0.11.19 /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends git libgomp1 && rm -rf /var/lib/apt/lists/*

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

# MSI is resolved from the private index at the version pinned in pyproject.toml.
ARG VALIDATION_PROFILE=none
ARG SYNTHEMA_PYPI_URL=https://pypi.synthema.rid-intrasoft.eu/simple/
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=secret,id=synthema_username,env=UV_INDEX_SYNTHEMA_USERNAME \
    --mount=type=secret,id=synthema_password,env=UV_INDEX_SYNTHEMA_PASSWORD \
    set -- --index "synthema=$SYNTHEMA_PYPI_URL"; \
    case "$VALIDATION_PROFILE" in \
      aml-full|scd-full) uv sync --locked --package synthema-fl-client --no-install-workspace "$@" --group dev --extra "$VALIDATION_PROFILE" ;; \
      none) uv sync --locked --package synthema-fl-client --no-install-workspace "$@" --group dev ;; \
      *) echo "Unsupported validation profile: $VALIDATION_PROFILE" >&2; exit 1 ;; \
    esac

# Require the MSI preprocessing API before producing a deployable image.
# Fail early if the published MSI package lacks the required API.
RUN python -c "from model_store_interface.step_zero import PlainStepZeroStrategy, StepZeroMetrics; from model_store_interface.metadata.models.type_constants import NUMERIC_TYPE_NAMES"

FROM ${BASE_IMAGE}:${CURRENT_VERSION} as runtime

ARG VALIDATION_PROFILE=none
ARG FIDELITY_ENABLED=false
ENV VALIDATION_PROFILE=${VALIDATION_PROFILE} FIDELITY_ENABLED=${FIDELITY_ENABLED} MPLBACKEND=Agg

WORKDIR /app

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/common:/app/src"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}
# grpcio-tools/protobuf come from the resolved application environment.
COPY apps/fl_client/src/ ./src
COPY apps/fl_client/tests/ ./tests

# Compile the protobuf file in the appropriate directory
RUN python -m grpc_tools.protoc -I/app/src/fl_client --python_out=/app/src/fl_client --grpc_python_out=/app/src/fl_client /app/src/fl_client/smpc.proto
RUN sed -i 's/^import smpc_pb2 as smpc__pb2$/import fl_client.smpc_pb2 as smpc__pb2/' /app/src/fl_client/smpc_pb2_grpc.py

RUN if [ "$VALIDATION_PROFILE" = "aml-full" ] || [ "$VALIDATION_PROFILE" = "scd-full" ]; then \
      python -c 'import os; from fl_client.validation_runner import validator; from interfaces.validation_policy import required_reports; disease=os.environ["VALIDATION_PROFILE"].split("-")[0]; [validator(kind, disease) for kind in required_reports()]'; fi

CMD ["flower-client-app", "--insecure", "src.fl_client.main:app"]
