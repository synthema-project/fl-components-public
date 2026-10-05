# syntax=docker/dockerfile:1.10
ARG CURRENT_VERSION=latest
ARG BASE_IMAGE=fl-common

FROM python:3.11-slim as builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends git libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY wheels/ /wheels/
RUN touch README.md

RUN uv venv .venv
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV UV_FIND_LINKS=/wheels

# Resolve MSI from the exact validated wheel, including a version newer than
# the compatibility baseline declared in pyproject.toml.
RUN python -c "from pathlib import Path; import json; w=list(Path('/wheels').glob('model_store_interface-*.whl')); assert len(w)==1; p=Path('pyproject.toml'); p.write_text(p.read_text()+'\n[tool.uv]\noverride-dependencies = '+json.dumps(['model-store-interface @ '+w[0].as_uri()])+'\n')"
ARG VALIDATION_PROFILE=none
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
    case "$VALIDATION_PROFILE" in \
      aml-full|scd-full) uv sync --no-install-project --group dev --extra "$VALIDATION_PROFILE" ;; \
      *) uv sync --no-install-project --group dev ;; \
    esac

# Reporting is intentionally isolated from training dependencies.
ARG VALIDATION_PROFILE=none
RUN mkdir -p /opt/aml-utility \
 && case "$VALIDATION_PROFILE" in \
      none|aml-full|scd-full) ;; \
      aml) set -- /wheels/model_store_interface-*.whl; uv venv /opt/aml-utility \
        && uv pip install --python /opt/aml-utility/bin/python \
          "$1[aml-utility]" /wheels/sd_quality-*.whl \
          "mlflow-skinny==2.22.5" "numpy==1.26.4" "pandas==2.3.3" \
          "scikit-learn==1.7.2" "scikit-survival==0.26.0" \
        && /opt/aml-utility/bin/python -c "from model_store_interface.aml_utility import generate_aml_utility_report; from sd_quality.report.Generate_report import generate_report_from_dataframes_and_info" ;; \
      *) echo "Unsupported validation profile: $VALIDATION_PROFILE" >&2; exit 1 ;; \
    esac

FROM ${BASE_IMAGE}:${CURRENT_VERSION} as runtime

ARG VALIDATION_PROFILE=none
ENV VALIDATION_PROFILE=${VALIDATION_PROFILE} MPLBACKEND=Agg
COPY --from=builder /opt/aml-utility /opt/aml-utility

WORKDIR /app

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/common:/app/src"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}
# grpcio-tools/protobuf come from the resolved application environment.
COPY src/ ./src
COPY tests/ ./tests

# Compile the protobuf file in the appropriate directory
RUN python -m grpc_tools.protoc -I/app/src/fl_client --python_out=/app/src/fl_client --grpc_python_out=/app/src/fl_client /app/src/fl_client/smpc.proto
RUN sed -i 's/^import smpc_pb2 as smpc__pb2$/import fl_client.smpc_pb2 as smpc__pb2/' /app/src/fl_client/smpc_pb2_grpc.py

RUN if [ "$VALIDATION_PROFILE" = "aml-full" ] || [ "$VALIDATION_PROFILE" = "scd-full" ]; then \
      python -c 'import os; from fl_client.validation_runner import validator; disease=os.environ["VALIDATION_PROFILE"].split("-")[0]; [validator(kind, disease) for kind in ("utility", "privacy", "fidelity")]'; fi

CMD ["flower-client-app", "--insecure", "src.fl_client.main:app"]