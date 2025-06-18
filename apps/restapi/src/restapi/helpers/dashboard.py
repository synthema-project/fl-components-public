from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException

from restapi.auth import keycloak, get_current_user, get_current_user_with_restricted_role
from restapi.helpers.mlflow import (
    create_user as mlflow_create_user,
    get_user as mlflow_get_user,
    update_password as mlflow_update_password,
    update_admin as mlflow_update_admin,
    delete_user as mlflow_delete_user
)

from restapi.routers.keycloak_admin import get_keycloak_user_by_username


# ------------------ SCHEMAS ------------------ #
class CreateUserRequest(BaseModel):
    username: str
    email: str
    firstName: str = ""
    lastName: str = ""
    enabled: bool = True
    password: str
    is_admin: bool = False  # New parameter


class AdminStatusUpdateRequest(BaseModel):
    username: str
    admin: bool


# --- Models ---
class MLFlowUserCreateRequest(BaseModel):
    username: str
    password: str


class UpdatePasswordRequest(BaseModel):
    username: str
    new_password: str


def user_exists_in_keycloak(username: str) -> bool:
    user = get_keycloak_user_by_username(username)
    return user is not None


def user_exists_in_mlflow(username: str) -> bool:
    response = mlflow_get_user(username)
    return "user_info" in response  # adjust if needed based on actual response


def create_user(
    data: CreateUserRequest,
    user=Depends(get_current_user_with_restricted_role(required_roles=["central_node_admin"]))
):
    # Check if user already exists in Keycloak
    # if user_exists_in_keycloak(data.username):
    #    raise HTTPException(status_code=400, detail=f"User '{data.username}' already exists in Keycloak")

    # Check if user already exists in MLflow
    #if user_exists_in_mlflow(data.username):
    #    raise HTTPException(status_code=400, detail=f"User '{data.username}' already exists in MLflow")

    # Create user in Keycloak
    new_user = keycloak.create_user(
        username=data.username,
        email=data.email,
        first_name=data.firstName,
        last_name=data.lastName,
        enabled=data.enabled,
        password=data.password
    )

    # Create user in MLflow using imported method
    mlflow_user = MLFlowUserCreateRequest(username=data.username, password=data.password)
    mlflow_create_user(mlflow_user)

    # Set admin status if requested
    if data.is_admin:
        set_keycloak_admin_status(data.username, make_admin=True)
        mlflow_update_admin({"username": data.username, "admin": True})

    return new_user


def delete_user(
    username: str,
    user=Depends(get_current_user_with_restricted_role(required_roles=["central_node_admin"]))
):
    # Delete from Keycloak
    user_obj = get_keycloak_user_by_username(username)
    if user_obj:
        keycloak.delete_user(user_obj["id"])
    else:
        print(f"User '{username}' not found in Keycloak")

    # Delete from MLflow using imported method
    mlflow_delete_user({"username": username})

    return {"message": f"User '{username}' deleted successfully"}


def update_user_password(
    data: UpdatePasswordRequest,
    user=Depends(get_current_user_with_restricted_role(required_roles=["central_node_admin"]))
):
    # Update in Keycloak
    keycloak.update_user_password(username=data.username, password=data.new_password)

    # Update in MLflow using imported method
    mlflow_update_password(data)

    return {"message": f"Password updated for user '{data.username}' in Keycloak and MLflow"}


def set_mlflow_admin_status(username: str, make_admin: bool):
    # Use imported update_admin method
    mlflow_update_admin({"username": username, "admin": make_admin})


def set_keycloak_admin_status(username: str, make_admin: bool):
    user_id = keycloak.get_user_id(username)
    if not user_id:
        raise HTTPException(status_code=404, detail=f"User '{username}' not found in Keycloak")

    realm_roles = keycloak.get_realm_roles()
    admin_role = next((r for r in realm_roles if r["name"] == "realm-admin"), None)
    if not admin_role:
        raise HTTPException(status_code=500, detail="Keycloak admin role not found")

    if make_admin:
        keycloak.assign_realm_roles(user_id=user_id, roles=[admin_role])
    else:
        keycloak.remove_realm_roles(user_id=user_id, roles=[admin_role])


def update_admin_status(
    data: AdminStatusUpdateRequest,
    user=Depends(get_current_user_with_restricted_role(required_roles=["central_node_admin"]))
):
    # Keycloak
    set_keycloak_admin_status(username=data.username, make_admin=data.admin)

    # MLflow using imported method
    mlflow_update_admin({"username": data.username, "admin": data.admin})

    return {
        "message": f"User '{data.username}' {'promoted to' if data.admin else 'demoted from'} admin in Keycloak and MLflow"
    }
