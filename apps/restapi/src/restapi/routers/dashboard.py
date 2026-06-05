from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from restapi.auth import UserClaims, require_authentication
from restapi.helpers.mlflow_helper import get_experiment_id_by_name, get_run_by_name, get_child_runs_by_name, get_full_model_info_by_name, get_model_versions_by_name, set_mlflow_admin_permission
from restapi.helpers.dashboard import CreateUserRequest, UpdatePasswordRequest
from restapi.helpers.dashboard import create_user, delete_user, update_user_password
from schemas.task import TaskRead
from restapi.db import get_session
from sqlalchemy.ext.asyncio import AsyncSession

from restapi.helpers.mlflow_helper import MLFLOW_BASE_URL
from restapi.helpers.mlflow_helperv2 import create_federated_job_fetcher

federated_job_fetcher = create_federated_job_fetcher(MLFLOW_BASE_URL)


router = APIRouter(prefix="/dashboard", tags=["dashboard"])

@router.get("/v2/model/<model_id>")
async def get_model_by_id(model_id: str, current_user: UserClaims = Depends(require_authentication)):
    model = federated_job_fetcher.get_models(model_id)
    return model

@router.get("/v2/models")
async def get_models(current_user: UserClaims = Depends(require_authentication)):
    models = federated_job_fetcher.get_models()
    return models

@router.get("/v2/task/<task_id>")
async def get_task_by_id_v2(task_id: str, current_user: UserClaims = Depends(require_authentication)):
    task = federated_job_fetcher.get_federated_job_info(task_id)
    return task

@router.get("/v2/tasks")
async def get_tasks_v2(current_user: UserClaims = Depends(require_authentication)):
    tasks = federated_job_fetcher.list_federated_jobs()
    return tasks

@router.get("/v2/run_metrics/<run_id>/<metric_name>")
async def get_run_metrics(run_id: str, metric_name: str, current_user: UserClaims = Depends(require_authentication)):
    metrics = federated_job_fetcher.fetch_run_metric(run_id, metric_name)
    return metrics


@router.get("/model/<model_name>")
async def get_model_by_name(model_name: str, current_user: UserClaims = Depends(require_authentication)):
    model = get_full_model_info_by_name(model_name, user)
    return model


@router.get("/task/<task_id>")
async def get_task_by_id_(task_id : int, session: AsyncSession = Depends(get_session), current_user: UserClaims = Depends(require_authentication)):
    task = await session.get(TaskRead, task_id)
    if not task:
        return JSONResponse(status_code=404, content={"message": "Task not found"})
    # task = await get_task_by_id(task_id, session, user)
    experiment_id = get_experiment_id_by_name(task.experiment_name, user)
    # experiment = get_experiment(experiment_id)
    parent_run = get_run_by_name(task.run_name, experiment_id)
    child_runs = get_child_runs_by_name(task.run_name, experiment_id)

    # Fetch all model versions
    all_model_versions = get_model_versions_by_name(task.model_name, user)
    # Filter the one used by this task
    matching_model_version = next(
        (v for v in all_model_versions if int(v["version"]) == int(task.model_version)),
        None  # Optional default if no match is found
    )


    task_data = task.dict()
    task_data["run"] = parent_run["run"]
    task_data["children_runs"] = child_runs["runs"]
    task_data["model"] = matching_model_version

    return task_data




@router.post("/create/user")
async def create_user_(data: CreateUserRequest, current_user: UserClaims = Depends(require_authentication)):
    new_user = create_user(data, current_user)
    return {"user": new_user}

@router.delete("/delete/user/{username}")
async def delete_user_(username: str, current_user: UserClaims = Depends(require_authentication)):
    delete_user(username, current_user)
    return {"message": f"User '{username}' deleted successfully "}

@router.put("/update/password")
async def update_user_password_(data: UpdatePasswordRequest, current_user: UserClaims = Depends(require_authentication)):
    update_user_password(data, current_user)
    return {"message": f"Password updated for user '{data.username}' in Keycloak and MLflow"}

@router.put("/role/admin-upgrade")
async def upgrade_user_to_admin(user_id: str, mlflow_username: str, current_user: UserClaims = Depends(require_authentication)):
    upgrade_user_to_admin_keycloak(user_id)
    set_mlflow_admin_permission(mlflow_username, level="ADMIN")
    return {"message": "User upgraded to admin in both Keycloak and MLflow"}


@router.put("/role/admin-downgrade")
def downgrade_user_from_admin(user_id: str, mlflow_username: str, current_user: UserClaims = Depends(require_authentication)):
    downgrade_user_from_admin_keycloak(user_id)
    set_mlflow_admin_permission(mlflow_username, level="READ")
    return {"message": "User downgraded from admin in both Keycloak and MLflow"}


@router.post("/logout")
def logout_(user_id: str, current_user: UserClaims = Depends(require_authentication)):
    return logout_user(user_id)





#@router.put("/update/admin")
#def update_admin_status(
#    data: AdminStatusUpdateRequest,
#    current_user: UserClaims = Depends(require_authentication)
#):
    # TODO

#    return {
#        "message": f"User '{data.username}' {'promoted to' if data.admin else 'demoted from'} admin in Keycloak and MLflow"
#    }