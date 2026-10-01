# Building env
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

ENV PYTHOPATH="${PYTHOPATH}:/common"
COPY interfaces/ /common/interfaces/
COPY schemas/ /common/schemas/
COPY fl_models/ /common/fl_models/
COPY testing/ /common/testing/
