"""Obtain a fresh catalogue service token, with legacy bearer-token support."""
import os
from urllib.parse import quote
import requests


def catalogue_headers(fallback=''):
    client_id = os.getenv('KEYCLOAK_CLIENT_ID')
    secret = os.getenv('KEYCLOAK_CLIENT_SECRET')
    if client_id and secret:
        base = os.environ['KEYCLOAK_SERVER_URL'].rstrip('/')
        realm = quote(os.environ['KEYCLOAK_REALM_NAME'], safe='')
        response = requests.post(f'{base}/realms/{realm}/protocol/openid-connect/token',
            data={'grant_type': 'client_credentials', 'client_id': client_id,
                  'client_secret': secret}, timeout=20)
        response.raise_for_status()
        return {'Authorization': 'Bearer ' + response.json()['access_token']}
    token = fallback or os.getenv('DATA_CATALOGUE_TOKEN', '')
    if not token:
        raise RuntimeError('Configure KEYCLOAK_CLIENT_ID and KEYCLOAK_CLIENT_SECRET or DATA_CATALOGUE_TOKEN')
    return {'Authorization': 'Bearer ' + token}
