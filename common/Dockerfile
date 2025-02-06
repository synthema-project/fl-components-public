# Building env
FROM python:3.11-slim

ENV PYTHOPATH="${PYTHOPATH}:/common"
COPY interfaces/ /common/interfaces/
COPY schemas/ /common/schemas/
COPY fl_models/ /common/fl_models/