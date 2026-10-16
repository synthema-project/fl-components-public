.PHONY: install lock test test-cov lint lint-fix format type-check run-fl-client run-fl-server run-superlink

ROOT := $(CURDIR)
export PYTHONPATH := $(ROOT)/common:$(ROOT)/apps/fl_client/src:$(ROOT)/apps/fl_server/src:$(ROOT)/apps/restapi/src:$(ROOT)/apps/status_listener/src:$(ROOT)
UV := uv run --locked

install:
	bash scripts/generate_envs.sh

lock:
	python3 scripts/lock_dependencies.py --update

test:
	$(UV) --directory common pytest interfaces schemas
	$(UV) --directory apps/fl_client pytest tests
	$(UV) --directory apps/fl_server pytest tests
	$(UV) --directory apps/restapi pytest tests
	$(UV) --directory apps/status_listener pytest tests
	$(UV) pytest scripts/tests

test-cov:
	$(UV) --all-packages pytest --import-mode=importlib --cov common apps scripts/tests

lint:
	$(UV) ruff check

lint-fix:
	$(UV) ruff check --fix

format:
	$(UV) ruff format

type-check:
	$(UV) --all-packages mypy apps/fl_client/src apps/fl_server/src common --explicit-package-bases

run-superlink:
	$(UV) --package synthema-fl-server flower-superlink --insecure

run-fl-server:
	$(UV) --package synthema-fl-server python -m fl_server.main

run-fl-client:
	$(UV) --package synthema-fl-client flower-client-app --insecure fl_client.main:app
