"""Service credential selection without contacting Keycloak."""
from unittest.mock import Mock

import pytest

from interfaces.catalogue_auth import catalogue_headers


@pytest.fixture(autouse=True)
def credentials(monkeypatch):
    for name in ('KEYCLOAK_CLIENT_ID', 'KEYCLOAK_CLIENT_SECRET', 'DATA_CATALOGUE_TOKEN'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('KEYCLOAK_SERVER_URL', 'https://identity.example/')
    monkeypatch.setenv('KEYCLOAK_REALM_NAME', 'test realm')


def test_service_token(monkeypatch):
    monkeypatch.setenv('KEYCLOAK_CLIENT_ID', 'shared')
    monkeypatch.setenv('KEYCLOAK_CLIENT_SECRET', 'shared-secret')
    response = Mock()
    response.json.return_value = {'access_token': 'fresh-token'}
    post = Mock(return_value=response)
    monkeypatch.setattr('interfaces.catalogue_auth.requests.post', post)
    assert catalogue_headers('old-token') == {'Authorization': 'Bearer fresh-token'}
    post.assert_called_once_with(
        'https://identity.example/realms/test%20realm/protocol/openid-connect/token',
        data={'grant_type': 'client_credentials', 'client_id': 'shared',
              'client_secret': 'shared-secret'}, timeout=20)
    response.raise_for_status.assert_called_once()


def test_legacy_token(monkeypatch):
    monkeypatch.setenv('DATA_CATALOGUE_TOKEN', 'legacy')
    assert catalogue_headers() == {'Authorization': 'Bearer legacy'}
    assert catalogue_headers('explicit') == {'Authorization': 'Bearer explicit'}


def test_missing_credentials():
    with pytest.raises(RuntimeError, match='KEYCLOAK_CLIENT_ID'):
        catalogue_headers()
