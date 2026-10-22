"""Keep unit tests independent of deployment credentials and disease selection."""
import os
import pytest
for name, value in {'MLFLOW_URL':'http://localhost', 'NODE_NAME':'ICH',
    'RABBIT_HOST':'localhost', 'RABBIT_PORT':'5672', 'RABBIT_USERNAME':'test',
    'RABBIT_PASSWORD':'test', 'DB_MODE':'sqlite', 'KEYCLOAK_SERVER_URL':'http://localhost/',
    'RABBIT_QUEUE':'status',
    'POSTGRES_USERNAME':'test', 'POSTGRES_PASSWORD':'test', 'POSTGRES_DB':'test',
    'POSTGRES_HOST':'localhost', 'POSTGRES_PORT':'5432',
    'KEYCLOAK_CLIENT_ID':'test', 'KEYCLOAK_REALM_NAME':'test',
    'DATA_CATALOGUE_URL':'http://catalogue', 'DATA_CATALOGUE_TOKEN':'test',
    'FLWR_TELEMETRY_ENABLED':'0'}.items():
    os.environ.setdefault(name, value)

@pytest.fixture(autouse=True)
def pool_only_by_default(monkeypatch):
    monkeypatch.setenv('VALIDATION_PROFILE', 'none')
