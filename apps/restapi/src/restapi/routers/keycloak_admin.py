from typing import Optional, List

from pydantic import BaseModel
from fastapi import APIRouter,FastAPI, HTTPException, Depends
from restapi.auth import keycloak, get_current_user, get_current_user_with_restricted_role, KEYCLOAK_SERVER_URL, oauth2_scheme, get_admin_token, get_group_members
import requests




# --------------  KEYCLOAK ADMIN ENDPOINTS --------------------- #

# ------------------ SCHEMAS ------------------ #
class CreateUserRequest(BaseModel):
    username: str
    email: str
    firstName: str = ""
    lastName: str = ""
    enabled: bool = True
    password: str

class CreateGroupRequest(BaseModel):
    name: str

class CreateRoleRequest(BaseModel):
    name: str

router = APIRouter(prefix="/keycloak", tags=["keycloak"])


# @router.get("/user")
# def get_user_by_query(query: str = None):
#     return keycloak.get_user(query=query)

@router.get("/admin/token")
def get_admin_token_(user = Depends(get_current_user_with_restricted_role(required_roles=["central_node_admin"]))):
    return get_admin_token()

@router.get("/user/{user_id}")
def get_user(user_id: str = None, user = Depends(get_current_user_with_restricted_role(required_roles=["central_node_admin"]))):
    return keycloak.get_user(user_id=user_id)

@router.get("/admin/users")
def list_users(user = Depends(get_current_user_with_restricted_role(required_roles=["central_node_admin"]))):
    return keycloak.get_all_users()

@router.post("/admin/users")
def create_user(data: CreateUserRequest, user = Depends(get_current_user_with_restricted_role(required_roles=["central_node_admin"]))):
    new_user = keycloak.create_user(
        username=data.username,
        email=data.email,
        first_name=data.firstName,
        last_name=data.lastName,
        enabled=data.enabled,
        # password= [{"value": data.password, "type": "password", "temporary": False}]
        password=data.password
    )
    return {"user_id": new_user}

# @router.put("/user")
# def update_user(user: KeycloakUser):
#     return keycloak.update_user(user=user)
#
# @router.put("/user/{user_id}/change-password")
# def change_password(user_id: str, new_password: SecretStr):
#     return keycloak.change_password(user_id=user_id, new_password=new_password)

# @router.put("/user/{user_id}/send-email-verification")
# def send_email_verification(user_id: str):
#     return keycloak.send_email_verification(user_id=user_id)

@router.delete("/admin/users/{user_id}")
def delete_user(user_id: str, user = Depends(get_current_user_with_restricted_role(required_roles=["central_node_admin"]))):
    keycloak.delete_user(user_id)
    return {"message": "User deleted"}

# ----- Group Endpoints -----
@router.get("/admin/groups")
def list_groups(user = Depends(get_current_user_with_restricted_role(required_roles=["central_node_admin"]))):
    return keycloak.get_all_groups()

# @router.get("/group/{group_id}")
def get_group(group_id: str, user = Depends(get_current_user_with_restricted_role(required_roles=["central_node_admin"]))):
    return keycloak.get_group(group_id)


@router.get("/group/name/{group_name}")
def get_group_by_name(group_name: str, user = Depends(get_current_user_with_restricted_role(required_roles=["central_node_admin"]))):
    return keycloak.get_groups([group_name])

## TODO
@router.get("/group/{group_id}")
def get_full_group(group_id: str, user = Depends(get_current_user_with_restricted_role(required_roles=["central_node_admin"]))):
    access_token = get_admin_token()["access_token"]
    url = f"{KEYCLOAK_SERVER_URL}/admin/realms/synthia/groups/{group_id}"

    response = requests.get(url, headers={
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    })
    print(response.json())

    response.raise_for_status()
    return response.json()

@router.get("/groups/{group_id}/members")
def read_group_members(group_id: str, brief: bool = True):
    try:
        members = get_group_members(group_id, brief_representation=brief)
        return members
    except requests.HTTPError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)

## TODO
@router.get("/groups/{group_id}/full")
def get_full_group_with_members(
    group_id: str,
    brief: bool = True,
    user=Depends(get_current_user_with_restricted_role(required_roles=["central_node_admin"]))
):
    try:
        # Get admin token
        access_token = get_admin_token()["access_token"]

        # Get group details
        group_url = f"{KEYCLOAK_SERVER_URL}/admin/realms/synthia/groups/{group_id}"
        group_response = requests.get(group_url, headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        })
        group_response.raise_for_status()
        group_data = group_response.json()

        # Get group members
        members = get_group_members(group_id, brief_representation=brief)

        # Combine and return
        return {
            "group": group_data,
            "members": members
        }

    except requests.HTTPError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)

@router.get("/users/{user_id}/groups")
def get_user_groups(
    user_id: str,
    brief: bool = False,
    user=Depends(get_current_user_with_restricted_role(required_roles=["central_node_admin"]))
):
    try:
        access_token = get_admin_token()["access_token"]
        url = f"{KEYCLOAK_SERVER_URL}/admin/realms/synthia/users/{user_id}/groups"
        params = {"briefRepresentation": str(brief).lower()}

        response = requests.get(url, headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }, params=params)

        response.raise_for_status()
        return response.json()

    except requests.HTTPError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)

# @router.get("/group-by-path/{path: path}")
# def get_group_by_path(path: str):
#     return keycloak.get_group_by_path(path)

@router.post("/admin/groups")
def create_group(data: CreateGroupRequest, user = Depends(get_current_user_with_restricted_role(required_roles=["central_node_admin"]))):
    keycloak.create_group({"name": data.name})
    return {"message": "Group created"}


@router.delete("/admin/groups/{group_id}")
def delete_group(group_id: str, user = Depends(get_current_user_with_restricted_role(required_roles=["central_node_admin"]))):
    keycloak.delete_group(group_id)
    return {"message": "Group deleted"}

# ----- Role Endpoints -----
@router.post("/admin/roles")
def create_role(data: CreateRoleRequest, user = Depends(get_current_user_with_restricted_role(required_roles=["central_node_admin"]))):
    keycloak.create_realm_role({"name": data.name})
    return {"message": "Role created"}

@router.get("/admin/roles")
def list_roles(user = Depends(get_current_user_with_restricted_role(required_roles=["central_node_admin"]))):
    return keycloak.get_realm_roles()

@router.delete("/admin/roles/{role_name}")
def delete_role(role_name: str, user = Depends(get_current_user_with_restricted_role(required_roles=["central_node_admin"]))):
    keycloak.delete_realm_role(role_name)
    return {"message": "Role deleted"}

@router.get("/roles")
def get_all_roles(user = Depends(get_current_user_with_restricted_role(required_roles=["central_node_admin"]))):
    return keycloak.get_all_roles()


# User Roles

# @router.post("/users/{user_id}/roles")
# def add_roles_to_user(user_id: str, roles: Optional[List[str]] = Query(None)):
#     return keycloak.add_user_roles(user_id=user_id, roles=roles)


@router.get("/users/{user_id}/roles")
def get_user_roles(user_id: str, user = Depends(get_current_user_with_restricted_role(required_roles=["central_node_admin"]))):
    return keycloak.get_user_roles(user_id=user_id)


# @router.delete("/users/{user_id}/roles")
# def delete_roles_from_user(user_id: str, roles: Optional[List[str]] = Query(None)):
#     return keycloak.remove_user_roles(user_id=user_id, roles=roles)


# User Groups

@router.post("/users/{user_id}/groups")
def add_group_to_user(user_id: str, group_id: str, user = Depends(get_current_user_with_restricted_role(required_roles=["central_node_admin"]))):
    return keycloak.add_user_group(user_id=user_id, group_id=group_id)


@router.get("/users/{user_id}/groups")
def get_user_groups(user_id: str, user = Depends(get_current_user_with_restricted_role(required_roles=["central_node_admin"]))):
    return keycloak.get_user_groups(user_id=user_id)


@router.delete("/users/{user_id}/groups")
def delete_groups_from_user(user_id: str, group_id: str, user = Depends(get_current_user_with_restricted_role(required_roles=["central_node_admin"]))):
    return keycloak.remove_user_group(user_id=user_id, group_id=group_id)


def get_keycloak_user_by_username(username: str, user=Depends(get_current_user)) -> object | None:
    users = keycloak.get_all_users()
    for user_obj in users:
        if getattr(user_obj, "username", None) == username:
            return user_obj
    return None