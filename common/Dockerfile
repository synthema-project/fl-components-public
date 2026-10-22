# Building env
FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:0.11.19 /uv /uvx /usr/local/bin/
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH="/common"
COPY common/interfaces/ /common/interfaces/
COPY common/schemas/ /common/schemas/
COPY common/fl_models/ /common/fl_models/
COPY common/testing/ /common/testing/
