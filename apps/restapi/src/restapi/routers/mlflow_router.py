from fastapi import APIRouter, HTTPException, Depends
from restapi.helpers.mlflow_helper import MLFlowUserCreateRequest, UpdatePasswordRequest, UpdateAdminRequest, PermissionRequest
from restapi.helpers.mlflow_helper import create_user, get_user, update_password, update_admin, delete_user, create_experiment_permission, get_experiment_permission, update_experiment_permission, delete_experiment_permission, create_model_permission, get_model_permission, update_model_permission, delete_model_permission
from restapi.helpers.mlflow_helper import search_experiments, get_experiment, create_experiment, delete_experiment, restore_experiment, search_runs, get_run, delete_run, restore_run, list_registered_models, get_registered_model, create_registered_model, delete_registered_model, get_model_version, get_model_versions, get_model_download_uri, get_all_run_metrics, get_run_metrics, search_model_versions_by_run
from restapi.auth import UserClaims, require_authentication

router_admin = APIRouter(prefix="/mlflow-admin", tags=["mlflow-admin"])


# ============================
#        USER MANAGEMENT
# ============================

@router_admin.post("/users")
def create_user_(new_user: MLFlowUserCreateRequest, current_user: UserClaims = Depends(require_authentication)):
    if not current_user.has_role("Admin"):
        raise HTTPException(status_code=403, detail="No permissions to access this endpoint")
    return create_user(new_user, user)

@router_admin.get("/users/{username}")
def get_user_(username: str, current_user: UserClaims = Depends(require_authentication)):
    return get_user(username, user)

@router_admin.post("/users/update-password")
def update_password_(payload: UpdatePasswordRequest, current_user: UserClaims = Depends(require_authentication)):
    if not current_user.has_role("Admin"):
        raise HTTPException(status_code=403, detail="No permissions to access this endpoint")
    return update_password(payload, user)

@router_admin.post("/users/update-admin")
def update_admin_(payload: UpdateAdminRequest, current_user: UserClaims = Depends(require_authentication)):
    if not current_user.has_role("Admin"):
        raise HTTPException(status_code=403, detail="No permissions to access this endpoint")
    return update_admin(payload, user)

@router_admin.post("/users/delete")
def delete_user_(username: str, current_user: UserClaims = Depends(require_authentication)):
    if not current_user.has_role("Admin"):
        raise HTTPException(status_code=403, detail="No permissions to access this endpoint")
    return delete_user(username, user)


# ============================
#   EXPERIMENT PERMISSIONS
# ============================

@router_admin.post("/permissions/experiments/create")
def create_experiment_permission_(payload: PermissionRequest, current_user: UserClaims = Depends(require_authentication)):
    if not current_user.has_role("Admin"):
        raise HTTPException(status_code=403, detail="No permissions to access this endpoint")
    return create_experiment_permission(payload, current_user)

@router_admin.get("/permissions/experiments/get")
def get_experiment_permission_(object_id: str, current_user: UserClaims = Depends(require_authentication)):
    return get_experiment_permission(object_id, current_user)

@router_admin.post("/permissions/experiments/update")
def update_experiment_permission_(payload: PermissionRequest, current_user: UserClaims = Depends(require_authentication)):
    if not current_user.has_role("Admin"):
        raise HTTPException(status_code=403, detail="No permissions to access this endpoint")
    return update_experiment_permission(payload, current_user)

@router_admin.post("/permissions/experiments/delete")
def delete_experiment_permission_(payload: PermissionRequest, current_user: UserClaims = Depends(require_authentication)):
    if not current_user.has_role("Admin"):
        raise HTTPException(status_code=403, detail="No permissions to access this endpoint")
    return delete_experiment_permission(payload, current_user)


# ============================
#  REGISTERED MODEL PERMISSIONS
# ============================

@router_admin.post("/permissions/registered-models/create")
def create_model_permission_(payload: PermissionRequest, current_user: UserClaims = Depends(require_authentication)):
    if not current_user.has_role("Admin"):
        raise HTTPException(status_code=403, detail="No permissions to access this endpoint")
    return create_model_permission(payload, current_user)

@router_admin.get("/permissions/registered-models/get")
def get_model_permission_(object_id: str, current_user: UserClaims = Depends(require_authentication)):
    return get_model_permission(object_id, current_user)

@router_admin.post("/permissions/registered-models/update")
def update_model_permission_(payload: PermissionRequest, current_user: UserClaims = Depends(require_authentication)):
    if not current_user.has_role("Admin"):
        raise HTTPException(status_code=403, detail="No permissions to access this endpoint")
    return update_model_permission(payload, current_user)

@router_admin.post("/permissions/registered-models/delete")
def delete_model_permission_(payload: PermissionRequest, current_user: UserClaims = Depends(require_authentication)):
    if not current_user.has_role("Admin"):
        raise HTTPException(status_code=403, detail="No permissions to access this endpoint")
    return delete_model_permission(payload, current_user)


###### END OF ADMIN ENDPOINTS   ######
router = APIRouter(prefix="/mlflow", tags=["mlflow"])

# --- Experiments ---
@router.post("/experiments/search")
def search_experiments_(payload: dict = {}, user: UserClaims = Depends(require_authentication)):
    return search_experiments(payload, user)

@router.get("/experiments/get")
def get_experiment_(experiment_id: str, user: UserClaims = Depends(require_authentication)):
    return get_experiment(experiment_id, user)

@router.post("/experiments/create")
def create_experiment_(payload: dict, user: UserClaims = Depends(require_authentication)):
    return create_experiment(payload, user)

@router.post("/experiments/delete")
def delete_experiment_(payload: dict, user: UserClaims = Depends(require_authentication)):
    return delete_experiment(payload, user)

@router.post("/experiments/restore")
def restore_experiment_(payload: dict, user: UserClaims = Depends(require_authentication)):
    return restore_experiment(payload, user)

# --- Runs ---
@router.post("/runs/search")
def search_runs_(payload: dict, user: UserClaims = Depends(require_authentication)):
    return search_runs(payload, user)

@router.get("/runs/get")
def get_run_(run_id: str, user: UserClaims = Depends(require_authentication)):
    return get_run(run_id, user)

@router.post("/runs/delete")
def delete_run_(payload: dict, user: UserClaims = Depends(require_authentication)):
    return delete_run(payload, user)

@router.post("/runs/restore")
def restore_run_(payload: dict, user: UserClaims = Depends(require_authentication)):
    return restore_run(payload, user)

@router.post("/runs/metrics")
def get_run_metric(run_id: str, key: str,  user: UserClaims = Depends(require_authentication)):
    return get_run_metrics(run_id,key, user)

@router.post("/runs/metrics/all")
def get_run_metrics(run_id: str, user: UserClaims = Depends(require_authentication)):
    return get_all_run_metrics(run_id, user)

# --- Models ---
@router.get("/registered-models/list")
def list_registered_models_(user: UserClaims = Depends(require_authentication)):
    return list_registered_models(user)

@router.get("/registered-models/get")
def get_registered_model_(name: str, user: UserClaims = Depends(require_authentication)):
    return get_registered_model(name, user)

@router.post("/registered-models/create")
def create_registered_model_(payload: dict, user: UserClaims = Depends(require_authentication)):
    return create_registered_model(payload, user)

@router.post("/registered-models/delete")
def delete_registered_model_(payload: dict, user: UserClaims = Depends(require_authentication)):
    return delete_registered_model(payload, user)

# --- Model Versions ---
@router.get("/model-version/get")
def get_model_version_(name: str, version: str, user: UserClaims = Depends(require_authentication)):
    return get_model_version(name, version, user)

@router.get("/model-versions/list")
def get_model_version_(name: str, user: UserClaims = Depends(require_authentication)):
    return get_model_versions(name, user)

@router.get("/model-versions/get-download-uri")
def get_model_download_uri_(name: str, version: str, user: UserClaims = Depends(require_authentication)):
    return get_model_download_uri(name, version, user)

@router.post("/model-versions/search-by-run")
def search_model_versions_by_run_(
    payload: dict,
    user: UserClaims = Depends(require_authentication)
):
    run_id = payload.get("run_id")

    if not run_id:
        raise HTTPException(status_code=400, detail="run_id is required")

    return search_model_versions_by_run(run_id, user)
