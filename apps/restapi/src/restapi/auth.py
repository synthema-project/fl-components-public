from fastapi import Depends, HTTPException, status, APIRouter
from fastapi.security import OAuth2AuthorizationCodeBearer
from restapi import config
from fastapi_keycloak import FastAPIKeycloak
import requests


# Keycloak Configuration

KEYCLOAK_SERVER_URL = config.KEYCLOAK_SERVER_URL
KEYCLOAK_CLIENT_ID = config.KEYCLOAK_CLIENT_ID
KEYCLOAK_CLIENT_SECRET = config.KEYCLOAK_CLIENT_SECRET
KEYCLOAK_REALM_NAME = config.KEYCLOAK_REALM_NAME
KEYCLOAK_REDIRECT_URI = config.KEYCLOAK_REDIRECT_URI
KEYCLOAK_ADMIN_CLIENT_SECRET = config.KEYCLOAK_ADMIN_CLIENT_SECRET
KEYCLOAK_AUTHORIZED_GROUP = config.KEYCLOAK_AUTHORIZED_GROUP
KEYCLOAK_AUTHORIZED_ROLE = config.KEYCLOAK_AUTHORIZED_ROLE

# Initialize Keycloak Authentication
keycloak = FastAPIKeycloak(
    server_url=KEYCLOAK_SERVER_URL,
    client_id=KEYCLOAK_CLIENT_ID,
    client_secret=KEYCLOAK_CLIENT_SECRET,
    admin_client_id="admin-client",
    admin_client_secret="pqEnP3qqjaTq7Ux1qEoZ6jjF93cH7qx6",
    realm=KEYCLOAK_REALM_NAME,
    callback_uri=KEYCLOAK_REDIRECT_URI
)

# OAuth2 Configuration
oauth2_scheme = OAuth2AuthorizationCodeBearer(
    tokenUrl=f"{KEYCLOAK_SERVER_URL}/realms/{KEYCLOAK_REALM_NAME}/protocol/openid-connect/token",
    authorizationUrl=f"{KEYCLOAK_SERVER_URL}/realms/{KEYCLOAK_REALM_NAME}/protocol/openid-connect/auth"
)
# Function to Verify Token
def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        user_info = keycloak.get_current_user()
        return user_info
    except Exception:
        raise HTTPException(status_code=401, detail="User unauthorized")

# Function to Verify Token
def get_current_user_with_restricted_role(token: str = Depends(oauth2_scheme), required_roles = ["ai_practitioner"]):
    try:
        # required_roles = "ai_practitioner"
        user_info = keycloak.get_current_user(required_roles=required_roles)
        # print(f"User info: {user_info}")
        return user_info
    except Exception as e:
        # print(f"Auth failed: {str(e)}")
        raise HTTPException(status_code=401, detail="User unauthorized, please verify you have the correct role")


def get_admin_token():
    token_url = f"{KEYCLOAK_SERVER_URL}/realms/{KEYCLOAK_REALM_NAME}/protocol/openid-connect/token"
    data = {
        "grant_type": "password",
        "client_id": "admin-cli",
        "username": "santibl",
        "password": "gatvgatv",
    }

    response = requests.post(token_url, data=data)
    if not response.ok:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get token from Keycloak: {response.text}"
        )
    return response.json()

# def get_admin_token():
#     url = f"{KEYCLOAK_SERVER_URL}/realms/{KEYCLOAK_REALM_NAME}/protocol/openid-connect/token"
#     data = {
#         "client_id": "admin-cli",
#         "client_secret": "pqEnP3qqjaTq7Ux1qEoZ6jjF93cH7qx6",
#         "grant_type": "client_credentials"
#     }
#     response = requests.post(url, data=data)
#     response.raise_for_status()
#     return response.json()


def logout_user(user_id: str):
    token = get_admin_token()
    url = f"{KEYCLOAK_SERVER_URL}/admin/realms/{KEYCLOAK_REALM_NAME}/users/{user_id}/logout"
    response = requests.post(url, headers={"Authorization": f"Bearer {token}"})
    response.raise_for_status()
    return {"message": "User logged out"}

def assign_realm_role_to_user(user_id: str, role_name: str):
    token = get_admin_token()

    # Get the role from realm
    role_url = f"{KEYCLOAK_SERVER_URL}/admin/realms/{KEYCLOAK_REALM_NAME}/roles/{role_name}"
    role = requests.get(role_url, headers={"Authorization": f"Bearer {token}"}).json()

    # Assign role
    assign_url = f"{KEYCLOAK_SERVER_URL}/admin/realms/{KEYCLOAK_REALM_NAME}/users/{user_id}/role-mappings/realm"
    response = requests.post(assign_url, json=[role], headers={"Authorization": f"Bearer {token}"})
    response.raise_for_status()
    return {"message": f"Role '{role_name}' assigned to user"}

def remove_realm_role_from_user(user_id: str, role_name: str):
    token = get_admin_token()

    # Get the role
    role_url = f"{KEYCLOAK_SERVER_URL}/admin/realms/{KEYCLOAK_REALM_NAME}/roles/{role_name}"
    role = requests.get(role_url, headers={"Authorization": f"Bearer {token}"}).json()

    # Remove role
    remove_url = f"{KEYCLOAK_SERVER_URL}/admin/realms/{KEYCLOAK_REALM_NAME}/users/{user_id}/role-mappings/realm"
    response = requests.delete(remove_url, json=[role], headers={"Authorization": f"Bearer {token}"})
    response.raise_for_status()
    return {"message": f"Role '{role_name}' removed from user"}

def update_user_names(user_id: str, first_name: str, last_name: str):
    token = get_admin_token()
    update_url = f"{KEYCLOAK_SERVER_URL}/admin/realms/{KEYCLOAK_REALM_NAME}/users/{user_id}"
    payload = {
        "firstName": first_name,
        "lastName": last_name
    }
    response = requests.put(update_url, json=payload, headers={"Authorization": f"Bearer {token}"})
    response.raise_for_status()
    return {"message": "User name updated"}

def upgrade_user_to_admin_keycloak(user_id: str):
    token = get_admin_token()

    # Get the 'admin' role object
    role = requests.get(
        f"{KEYCLOAK_SERVER_URL}/admin/realms/{KEYCLOAK_REALM_NAME}/roles/admin",
        headers={"Authorization": f"Bearer {token}"}
    ).json()

    # Assign role
    assign_url = f"{KEYCLOAK_SERVER_URL}/admin/realms/{KEYCLOAK_REALM_NAME}/users/{user_id}/role-mappings/realm"
    resp = requests.post(assign_url, json=[role], headers={"Authorization": f"Bearer {token}"})
    resp.raise_for_status()
    return {"message": f"User {user_id} upgraded to admin in Keycloak"}

def downgrade_user_from_admin_keycloak(user_id: str):
    token = get_admin_token()

    role = requests.get(
        f"{KEYCLOAK_SERVER_URL}/admin/realms/{KEYCLOAK_REALM_NAME}/roles/admin",
        headers={"Authorization": f"Bearer {token}"}
    ).json()

    remove_url = f"{KEYCLOAK_SERVER_URL}/admin/realms/{KEYCLOAK_REALM_NAME}/users/{user_id}/role-mappings/realm"
    resp = requests.delete(remove_url, json=[role], headers={"Authorization": f"Bearer {token}"})
    resp.raise_for_status()
    return {"message": f"User {user_id} downgraded from admin in Keycloak"}


def get_group_members(group_id: str, brief_representation: bool = False):
    token = get_admin_token()

    url = (
        f"{KEYCLOAK_SERVER_URL}/admin/realms/{KEYCLOAK_REALM_NAME}/groups/{group_id}/members"
        f"?briefRepresentation={'true' if brief_representation else 'false'}"
    )

    response = requests.get(url, headers={"Authorization": f"Bearer {token}"})
    response.raise_for_status()

    return response.json()


