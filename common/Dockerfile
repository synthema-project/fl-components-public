# Building env
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH="${PYTHONPATH}:/common"
COPY common/interfaces/ /common/interfaces/
COPY common/schemas/ /common/schemas/
COPY common/fl_models/ /common/fl_models/
COPY common/testing/ /common/testing/
