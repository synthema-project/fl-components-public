"""Regression coverage for the environment-controlled authentication bypass."""
import asyncio
import importlib.util
from pathlib import Path
import sys

import pytest
from fastapi import HTTPException


@pytest.mark.parametrize(
    'value, bypass',
    [(None, False), ('false', False), ('False', False), ('0', False),
     ('no', False), ('off', False), ('', False), ('true', True),
     ('1', True), ('yes', True), (' ON ', True)],
)
def test_auth_bypass_requires_explicit_true(monkeypatch, value, bypass):
    for name, setting in {
        'RABBIT_USERNAME': 'test', 'RABBIT_PASSWORD': 'test',
        'RABBIT_HOST': 'localhost', 'RABBIT_PORT': '5672', 'DB_MODE': 'sqlite',
        'KEYCLOAK_SERVER_URL': 'http://localhost/', 'KEYCLOAK_CLIENT_ID': 'test',
        'KEYCLOAK_REALM_NAME': 'test',
    }.items():
        monkeypatch.setenv(name, setting)
    if value is None:
        monkeypatch.delenv('AUTH_DISABLE_TESTS', raising=False)
    else:
        monkeypatch.setenv('AUTH_DISABLE_TESTS', value)
    monkeypatch.setenv('AUTH_DISABLE_VERIFICATION', 'false')
    source = Path(__file__).parents[1] / 'src' / 'restapi'

    def load(name, filename):
        spec = importlib.util.spec_from_file_location(name, source / filename)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    config = load('restapi.config', 'config.py')
    auth = load('auth_regression', 'auth.py')
    assert config.AUTH_DISABLE_TESTS is bypass
    if bypass:
        assert asyncio.run(auth.get_current_user(None)).sub == 'test-user'
    else:
        with pytest.raises(HTTPException) as error:
            asyncio.run(auth.get_current_user(None))
        assert error.value.status_code == 401
        assert error.value.headers == {'WWW-Authenticate': 'Bearer'}
