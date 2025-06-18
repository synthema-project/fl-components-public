from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from restapi.auth import get_current_user, get_current_user_with_restricted_role
from requests.auth import HTTPBasicAuth
import requests

# --- Configuration ---
MLFLOW_BASE_URL = "http://mlflow-tracking.synthia.svc.cluster.local:80"
MLFLOW_USERNAME = "mlflow"
MLFLOW_PASSWORD = "mlflowpassword"

def get_auth():
    return HTTPBasicAuth(MLFLOW_USERNAME, MLFLOW_PASSWORD)


# --- Models ---
class MLFlowUserCreateRequest(BaseModel):
    username: str
    password: str

class UpdatePasswordRequest(BaseModel):
    username: str
    password: str

class UpdateAdminRequest(BaseModel):
    username: str
    is_admin: bool

class PermissionRequest(BaseModel):
    object_id: str
    username: str
    permission_level: str

class ObjectIdRequest(BaseModel):
    object_id: str

# --- Helper ---
def safe_http_request(method: str, endpoint: str, **kwargs):
    url = f"{MLFLOW_BASE_URL}{endpoint}"
    try:
        response = requests.request(method, url, auth=get_auth(), **kwargs)
        if response.status_code >= 400:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        return response.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================
#        USER MANAGEMENT
# ============================

def create_user(new_user: MLFlowUserCreateRequest, user=Depends(get_current_user_with_restricted_role(["central_node_admin"]))):
    return safe_http_request("POST", "/api/2.0/mlflow/users/create", json=new_user.dict())

def get_user(username: str, user=Depends(get_current_user)):
    return safe_http_request("GET", "/api/2.0/mlflow/users/get", params={"username": username})

def update_password(payload: UpdatePasswordRequest, user=Depends(get_current_user_with_restricted_role(["central_node_admin"]))):
    return safe_http_request("POST", "/api/2.0/mlflow/users/update-password", json=payload.dict())

def update_admin(payload: UpdateAdminRequest, user=Depends(get_current_user_with_restricted_role(["central_node_admin"]))):
    return safe_http_request("POST", "/api/2.0/mlflow/users/update-admin", json=payload.dict())

def get_model_permissions(username: str):
    # Example API call to get all model permissions for user
    response = safe_http_request(
        "GET",
        "/api/2.0/mlflow/permissions/model/list",
        params={"username": username}
    )
    return response.get("permissions", [])

def delete_model_permission(object_id: str, username: str):
    payload = {"object_id": object_id, "username": username}
    safe_http_request(
        "DELETE",
        "/api/2.0/mlflow/permissions/model/delete",
        json=payload
    )

def get_experiment_permissions(username: str):
    # Example API call to get all experiment permissions for user
    response = safe_http_request(
        "GET",
        "/api/2.0/mlflow/permissions/experiment/list",
        params={"username": username}
    )
    return response.get("permissions", [])

def delete_experiment_permission(object_id: str, username: str):
    payload = {"object_id": object_id, "username": username}
    safe_http_request(
        "DELETE",
        "/api/2.0/mlflow/permissions/experiment/delete",
        json=payload
    )


def delete_user(payload: dict, user=Depends(get_current_user_with_restricted_role(["central_node_admin"]))):
    username = payload.get("username")
    if not username:
        raise HTTPException(status_code=400, detail="Username is required for deletion")

    # 1. Delete all model permissions for the user
    for perm in get_model_permissions(username):
        delete_model_permission(perm["object_id"], username)

    # 2. Delete all experiment permissions for the user
    for perm in get_experiment_permissions(username):
        delete_experiment_permission(perm["object_id"], username)

    # 3. Delete user form MLFlow
    return safe_http_request("POST", "/api/2.0/mlflow/users/delete", json=payload)

# ============================
#   EXPERIMENT PERMISSIONS
# ============================

def create_experiment_permission(payload: PermissionRequest, user=Depends(get_current_user_with_restricted_role(["central_node_admin"]))):
    return safe_http_request("POST", "/api/2.0/mlflow/permissions/experiments/create", json=payload.dict())

def get_experiment_permission(object_id: str, user=Depends(get_current_user)):
    return safe_http_request("GET", "/api/2.0/mlflow/permissions/experiments/get", params={"object_id": object_id})

def update_experiment_permission(payload: PermissionRequest, user=Depends(get_current_user_with_restricted_role(["central_node_admin"]))):
    return safe_http_request("POST", "/api/2.0/mlflow/permissions/experiments/update", json=payload.dict())

# def delete_experiment_permission(payload: PermissionRequest, user=Depends(get_current_user_with_restricted_role(["central_node_admin"]))):
#    return safe_http_request("POST", "/api/2.0/mlflow/permissions/experiments/delete", json=payload.dict())


# ============================
#  REGISTERED MODEL PERMISSIONS
# ============================

def create_model_permission(payload: PermissionRequest, user=Depends(get_current_user_with_restricted_role(["central_node_admin"]))):
    return safe_http_request("POST", "/api/2.0/mlflow/permissions/registered-models/create", json=payload.dict())

def get_model_permission(object_id: str, user=Depends(get_current_user)):
    return safe_http_request("GET", "/api/2.0/mlflow/permissions/registered-models/get", params={"object_id": object_id})

def update_model_permission(payload: PermissionRequest, user=Depends(get_current_user_with_restricted_role(["central_node_admin"]))):
    return safe_http_request("POST", "/api/2.0/mlflow/permissions/registered-models/update", json=payload.dict())

#def delete_model_permission(payload: PermissionRequest, user=Depends(get_current_user_with_restricted_role(["central_node_admin"]))):
#    return safe_http_request("POST", "/api/2.0/mlflow/permissions/registered-models/delete", json=payload.dict())


###### END OF ADMIN ENDPOINTS   ######

# --- Experiments ---
def search_experiments(payload: dict = {}, user: dict = Depends(get_current_user)):
    return safe_http_request("POST", "/api/2.0/mlflow/experiments/search", json=payload)

def get_experiment(experiment_id: str, user: dict = Depends(get_current_user)):
    return safe_http_request("GET", "/api/2.0/mlflow/experiments/get", params={"experiment_id": experiment_id})

def get_experiment_id_by_name(experiment_name: str, user: dict = Depends(get_current_user)):
    response = safe_http_request(
        "GET",
        "/api/2.0/mlflow/experiments/get-by-name",
        params={"experiment_name": experiment_name}
    )
    return response.get("experiment", {}).get("experiment_id")

def create_experiment(payload: dict, user: dict = Depends(get_current_user)):
    return safe_http_request("POST", "/api/2.0/mlflow/experiments/create", json=payload)

def delete_experiment(payload: dict, user: dict = Depends(get_current_user)):
    return safe_http_request("POST", "/api/2.0/mlflow/experiments/delete", json=payload)

def restore_experiment(payload: dict, user: dict = Depends(get_current_user)):
    return safe_http_request("POST", "/api/2.0/mlflow/experiments/restore", json=payload)

# --- Runs ---
def search_runs(payload: dict, user: dict = Depends(get_current_user)):
    return safe_http_request("POST", "/api/2.0/mlflow/runs/search", json=payload)

def get_run(run_id: str, user: dict = Depends(get_current_user)):
    return safe_http_request("GET", "/api/2.0/mlflow/runs/get", params={"run_id": run_id})

def delete_run(payload: dict, user: dict = Depends(get_current_user)):
    return safe_http_request("POST", "/api/2.0/mlflow/runs/delete", json=payload)

def restore_run(payload: dict, user: dict = Depends(get_current_user)):
    return safe_http_request("POST", "/api/2.0/mlflow/runs/restore", json=payload)


# Get child runs of a given parent run
def get_child_runs(parent_run_id: str, experiment_id: str, user: dict = Depends(get_current_user)):
    payload = {
        "experiment_ids": [experiment_id],
        "filter": f"tags.mlflow.parentRunId = '{parent_run_id}'",
        "max_results": 1000  # adjust as needed
    }
    return safe_http_request("POST", "/api/2.0/mlflow/runs/search", json=payload)

def get_run_id_by_name(run_name: str, experiment_id: str, user: dict = Depends(get_current_user)):
    payload = {
        "experiment_ids": [experiment_id],
        "filter": f"tags.mlflow.runName = '{run_name}'",
        "max_results": 1
    }
    response = safe_http_request("POST", "/api/2.0/mlflow/runs/search", json=payload)
    runs = response.get("runs", [])
    return runs[0]["info"]["run_id"] if runs else None

# Get run details by run name
def get_run_by_name(run_name: str, experiment_id: str, user: dict = Depends(get_current_user)):
    run_id = get_run_id_by_name(run_name, experiment_id, user)
    if run_id:
        return get_run(run_id, user)
    return {"error": "Run not found"}

# Get child runs by run name
def get_child_runs_by_name(run_name: str, experiment_id: str, user: dict = Depends(get_current_user)):
    run_id = get_run_id_by_name(run_name, experiment_id, user)
    if run_id:
        return get_child_runs(run_id, experiment_id, user)
    return {"error": "Run not found"}

# --- Models ---
def list_registered_models(user: dict = Depends(get_current_user)):
    return safe_http_request("GET", "/api/2.0/mlflow/registered-models/search")

def get_registered_model(name: str, user: dict = Depends(get_current_user)):
    return safe_http_request("GET", "/api/2.0/mlflow/registered-models/get", params={"name": name})

def create_registered_model(payload: dict, user: dict = Depends(get_current_user)):
    return safe_http_request("POST", "/api/2.0/mlflow/registered-models/create", json=payload)

def delete_registered_model(payload: dict, user: dict = Depends(get_current_user)):
    return safe_http_request("POST", "/api/2.0/mlflow/registered-models/delete", json=payload)

# --- Model Versions ---
def get_model_version(name: str, version: str, user: dict = Depends(get_current_user)):
    return safe_http_request("GET", "/api/2.0/mlflow/model-versions/search", params={"name": name, "version": version})

def get_model_download_uri(name: str, version: str, user: dict = Depends(get_current_user)):
    return safe_http_request("GET", "/api/2.0/mlflow/model-versions/get-download-uri", params={"name": name, "version": version})

def search_model_versions(
    model_name: str,
    user: dict = Depends(get_current_user)
):
    payload = {
        "filter": f"name='{model_name}'",
        "max_results": 1000
    }
    response = safe_http_request("GET", "/api/2.0/mlflow/model-versions/search", json=payload)
    return response.get("model_versions", [])

def get_model_versions_by_run_id(run_id: str, user: dict = Depends(get_current_user)):
    return search_model_versions(f"run_id = '{run_id}'", user)


def get_model_versions_by_name(name: str, user: dict = Depends(get_current_user)):
    return search_model_versions(name, user)

def get_latest_model_versions(name: str, stages: list[str] = None, user: dict = Depends(get_current_user)):
    payload = {"name": name}
    if stages:
        payload["stages"] = stages
    return safe_http_request("POST", "/api/2.0/mlflow/registered-models/get-latest-versions", json=payload)


def get_full_model_info_by_name(model_name: str, user: dict = Depends(get_current_user)):
    # Optionally retrieve the registered model info
    registered_model = get_registered_model(model_name, user)



    # Get latest versions by stage (e.g. Production, Staging, etc.)
    latest_versions_response = get_latest_model_versions(model_name, user=user)
    latest_versions = latest_versions_response["model_versions"]

    # Get all model versions
    model_versions = get_model_versions_by_name(model_name, user)

    # Map of run_id to run object
    run_map = {}
    for version in model_versions:
        run_id = version["run_id"]
        if run_id:
            run = get_run(run_id, user)
            if run:
                run_map[run_id] = run

    # Nest the run info into the corresponding model version
    enriched_versions = []
    for version in model_versions:
        run_id = version["run_id"]
        version_with_run = version.copy()
        if run_id and run_id in run_map:
            version_with_run["source_run"] = run_map[run_id]["run"]
        enriched_versions.append(version_with_run)

    return {
        "model": {
            **registered_model["registered_model"],
            "versions": enriched_versions,
            "latest_versions": latest_versions
        }
    }

def set_mlflow_admin_permission(mlflow_username: str, level: str = "ADMIN"):
    payload = {
        "user_id": mlflow_username,
        "permission_level": level  # "ADMIN", "EDIT", or "READ"
    }

    return safe_http_request(
        method="POST",
        endpoint="/api/2.0/mlflow/permissions/set-permissions",
        json=payload
    )