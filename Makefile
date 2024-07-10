.PHONY: install clean test test-cov lint lint-fix format type-check run-fl-client run-fl-server run-superlink

install:
	poetry install

test:
	poetry run pytest common/utils/tests fl_client/tests fl_server/tests

test-cov:
	poetry run pytest --cov

lint:
	poetry run ruff check

lint-fix:
	poetry run ruff check --fix

format:
	poetry run ruff format

type-check:
	poetry run mypy fl_client/src fl_server/src common --explicit-package-bases

run-superlink:
	poetry run flower-superlink --insecure

run-fl-server:
	poetry run python fl_server/src/main.py

run-fl-client:
	poetry run flower-client-app --insecure fl_client.src.main:app
