from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from restapi.auth import get_current_user, get_current_user_with_restricted_role
from requests.auth import HTTPBasicAuth
import requests
from restapi.helpers.mlflow import MLFlowUserCreateRequest, UpdatePasswordRequest, UpdateAdminRequest, PermissionRequest, ObjectIdRequest, safe_http_request
from restapi.helpers.mlflow import create_user, get_user, update_password, update_admin, delete_user, create_experiment_permission, get_experiment_permission, update_experiment_permission, delete_experiment_permission, create_model_permission, get_model_permission, update_model_permission, delete_model_permission
from restapi.helpers.mlflow import search_experiments, get_experiment, create_experiment, delete_experiment, restore_experiment, search_runs, get_run, delete_run, restore_run, list_registered_models, get_registered_model, create_registered_model, delete_registered_model, get_model_version, get_model_download_uri


router_admin = APIRouter(prefix="/mlflow-admin", tags=["mlflow-admin"])


# ============================
#        USER MANAGEMENT
# ============================

@router_admin.post("/users")
def create_user_(new_user: MLFlowUserCreateRequest, user=Depends(get_current_user_with_restricted_role(["central_node_admin"]))):
    return create_user(new_user, user)

@router_admin.get("/users/{username}")
def get_user_(username: str, user=Depends(get_current_user)):
    return get_user(username, user)

@router_admin.post("/users/update-password")
def update_password_(payload: UpdatePasswordRequest, user=Depends(get_current_user_with_restricted_role(["central_node_admin"]))):
    return update_password(payload, user)

@router_admin.post("/users/update-admin")
def update_admin_(payload: UpdateAdminRequest, user=Depends(get_current_user_with_restricted_role(["central_node_admin"]))):
    return update_admin(payload, user)

@router_admin.post("/users/delete")
def delete_user_(payload: dict, user=Depends(get_current_user_with_restricted_role(["central_node_admin"]))):
    return delete_user(payload, user)


# ============================
#   EXPERIMENT PERMISSIONS
# ============================

@router_admin.post("/permissions/experiments/create")
def create_experiment_permission_(payload: PermissionRequest, user=Depends(get_current_user_with_restricted_role(["central_node_admin"]))):
    return create_experiment_permission(payload, user)

@router_admin.get("/permissions/experiments/get")
def get_experiment_permission_(object_id: str, user=Depends(get_current_user)):
    return get_experiment_permission(object_id, user)

@router_admin.post("/permissions/experiments/update")
def update_experiment_permission_(payload: PermissionRequest, user=Depends(get_current_user_with_restricted_role(["central_node_admin"]))):
    return update_experiment_permission(payload, user)

@router_admin.post("/permissions/experiments/delete")
def delete_experiment_permission_(payload: PermissionRequest, user=Depends(get_current_user_with_restricted_role(["central_node_admin"]))):
    return delete_experiment_permission(payload, user)


# ============================
#  REGISTERED MODEL PERMISSIONS
# ============================

@router_admin.post("/permissions/registered-models/create")
def create_model_permission_(payload: PermissionRequest, user=Depends(get_current_user_with_restricted_role(["central_node_admin"]))):
    return create_model_permission(payload, user)

@router_admin.get("/permissions/registered-models/get")
def get_model_permission_(object_id: str, user=Depends(get_current_user)):
    return get_model_permission(object_id, user)

@router_admin.post("/permissions/registered-models/update")
def update_model_permission_(payload: PermissionRequest, user=Depends(get_current_user_with_restricted_role(["central_node_admin"]))):
    return update_model_permission(payload, user)

@router_admin.post("/permissions/registered-models/delete")
def delete_model_permission_(payload: PermissionRequest, user=Depends(get_current_user_with_restricted_role(["central_node_admin"]))):
    return delete_model_permission(payload, user)


###### END OF ADMIN ENDPOINTS   ######
router = APIRouter(prefix="/mlflow", tags=["mlflow"])

# --- Experiments ---
@router.post("/experiments/search")
def search_experiments_(payload: dict = {}, user: dict = Depends(get_current_user)):
    return search_experiments(payload, user)

@router.get("/experiments/get")
def get_experiment_(experiment_id: str, user: dict = Depends(get_current_user)):
    return get_experiment(experiment_id, user)

@router.post("/experiments/create")
def create_experiment_(payload: dict, user: dict = Depends(get_current_user)):
    return create_experiment(payload, user)

@router.post("/experiments/delete")
def delete_experiment_(payload: dict, user: dict = Depends(get_current_user)):
    return delete_experiment(payload, user)

@router.post("/experiments/restore")
def restore_experiment_(payload: dict, user: dict = Depends(get_current_user)):
    return restore_experiment(payload, user)

# --- Runs ---
@router.post("/runs/search")
def search_runs_(payload: dict, user: dict = Depends(get_current_user)):
    return search_runs(payload, user)

@router.get("/runs/get")
def get_run_(run_id: str, user: dict = Depends(get_current_user)):
    return get_run(run_id, user)

@router.post("/runs/delete")
def delete_run_(payload: dict, user: dict = Depends(get_current_user)):
    return delete_run(payload, user)

@router.post("/runs/restore")
def restore_run_(payload: dict, user: dict = Depends(get_current_user)):
    return restore_run(payload, user)

# --- Models ---
@router.get("/registered-models/list")
def list_registered_models_(user: dict = Depends(get_current_user)):
    return list_registered_models(user)

@router.get("/registered-models/get")
def get_registered_model_(name: str, user: dict = Depends(get_current_user)):
    return get_registered_model(name, user)

@router.post("/registered-models/create")
def create_registered_model_(payload: dict, user: dict = Depends(get_current_user)):
    return create_registered_model(payload, user)

@router.post("/registered-models/delete")
def delete_registered_model_(payload: dict, user: dict = Depends(get_current_user)):
    return delete_registered_model(payload, user)

# --- Model Versions ---
@router.get("/model-versions/get")
def get_model_version_(name: str, version: str, user: dict = Depends(get_current_user)):
    return get_model_version(name, version, user)

@router.get("/model-versions/get-download-uri")
def get_model_download_uri_(name: str, version: str, user: dict = Depends(get_current_user)):
    return get_model_download_uri(name, version, user)
