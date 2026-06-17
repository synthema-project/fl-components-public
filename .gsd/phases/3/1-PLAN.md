---
phase: 3
plan: 1
wave: 1
---

# Plan 3.1: Propagate DCAT-AP metadata to FL Client

## Objective
Propagate the `dataset_meta_json` field through the server/client Flower integration so that the client has it available in `global_vars` for future Phase validation steps.

## Context
- .gsd/SPEC.md
- .gsd/ROADMAP.md
- apps/fl_server/src/fl_server/app.py
- apps/fl_server/src/fl_server/requires.py
- common/interfaces/recordset.py
- apps/fl_client/src/fl_client/interface.py
- apps/fl_client/src/fl_client/stages.py

## Tasks

<task type="auto">
  <name>Update common recordset configuration</name>
  <files>
    common/interfaces/recordset.py
  </files>
  <action>
    - Update `create_set_run_cfg_recordset` and `read_set_run_cfg_recordset` to accept and extract `dataset_meta_json: str | None = None`.
    - If `dataset_meta_json` is not None, add it to the `ConfigsRecord` dict. (Note: `ConfigsRecord` does not allow `None` values, so only insert if it exists).
    - In `read_set_run_cfg_recordset`, retrieve it safely via `.get("dataset_meta_json")` and return it in the result dict.
  </action>
  <verify>uv run pytest apps/fl_server/tests/ -v</verify>
  <done>RecordSet correctly serializes and deserializes the optional `dataset_meta_json` string.</done>
</task>

<task type="auto">
  <name>Propagate from server to FL message</name>
  <files>
    apps/fl_server/src/fl_server/app.py
    apps/fl_server/src/fl_server/requires.py
  </files>
  <action>
    - In `app.py::configure_mlflow`, pass `task.dataset_meta_json` as the final argument to `requires.set_run_config`.
    - In `requires.py::set_run_config`, update signature to accept `dataset_meta_json: str | None = None` and pass it to `create_set_run_cfg_recordset`.
  </action>
  <verify>uv run pytest apps/fl_server/tests/ -v</verify>
  <done>The server successfully includes `dataset_meta_json` in the configuration dispatch.</done>
</task>

<task type="auto">
  <name>Deserialize on client and store in global_vars</name>
  <files>
    apps/fl_client/src/fl_client/interface.py
    apps/fl_client/src/fl_client/stages.py
  </files>
  <action>
    - In `interface.py::set_run_config_if`, read `dataset_meta_json` from the `content` dict and pass it to `stages.set_run_config`.
    - In `stages.py::set_run_config`, update signature to accept `dataset_meta_json: str | None = None`. Store the raw string in `global_vars["dataset_meta_json"]`.
    - Also in `stages.py::set_run_config`, import `json` and use `json.loads(dataset_meta_json)` to store the dict in `global_vars["dataset_meta"]` if it is not None (otherwise store `None`). 
  </action>
  <verify>uv run pytest apps/fl_client/tests/ -v</verify>
  <done>The client successfully deserializes the metadata and it is available in `global_vars["dataset_meta"]`.</done>
</task>

## Success Criteria
- [ ] `dataset_meta_json` correctly passes from server RabbitMQ message to client Flower global variables.
- [ ] Tests pass across server, common, and client apps.
