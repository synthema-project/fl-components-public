"""Obtain catalogue tokens using a dedicated user or service-client credentials."""
import os
from urllib.parse import quote
import requests


def catalogue_headers(fallback=''):
    mode = os.getenv('KEYCLOAK_AUTH_MODE', 'client_credentials')
    client_id = os.getenv('KEYCLOAK_CLIENT_ID')
    secret = os.getenv('KEYCLOAK_CLIENT_SECRET')
    if mode == 'password':
        username = os.getenv('KEYCLOAK_USERNAME')
        password = os.getenv('KEYCLOAK_PASSWORD')
        if not all((client_id, username, password)):
            raise RuntimeError('Password authentication requires KEYCLOAK_CLIENT_ID, '
                               'KEYCLOAK_USERNAME and KEYCLOAK_PASSWORD')
        data = {'grant_type': 'password', 'client_id': client_id,
                'username': username, 'password': password}
        if secret:
            data['client_secret'] = secret
    elif mode == 'client_credentials':
        if not (client_id and secret):
            # Backward compatibility for existing deployments outside chart 2.x.
            token = fallback or os.getenv('DATA_CATALOGUE_TOKEN', '')
            if not token:
                raise RuntimeError('Configure KEYCLOAK_CLIENT_ID and KEYCLOAK_CLIENT_SECRET '
                                   'or select password authentication with dedicated user credentials')
            return {'Authorization': 'Bearer ' + token}
        data = {'grant_type': 'client_credentials', 'client_id': client_id,
                'client_secret': secret}
    else:
        raise RuntimeError('KEYCLOAK_AUTH_MODE must be password or client_credentials')
    base = os.environ['KEYCLOAK_SERVER_URL'].rstrip('/')
    realm = quote(os.environ['KEYCLOAK_REALM_NAME'], safe='')
    response = requests.post(f'{base}/realms/{realm}/protocol/openid-connect/token',
                             data=data, timeout=20)
    response.raise_for_status()
    return {'Authorization': 'Bearer ' + response.json()['access_token']}
