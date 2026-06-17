---
phase: 4
plan: 2
wave: 2
---

# Plan 4.2: Server-Side Post-Training Orchestration

## Objective
Add post-training synthetic data generation, federated validation, MLflow artifact logging,
and catalogue publication to `fl_server/app.py::server_main`.

## Context
- .gsd/ROADMAP.md
- apps/fl_server/src/fl_server/app.py
- apps/fl_server/src/fl_server/requires.py
- apps/fl_server/src/fl_server/interfaces/mlflow_client.py
- common/interfaces/recordset.py (plan 4.1 extensions must be complete)
- model_store_interface.catalogue_hook.publish_to_catalogue
- model_store_interface.validation (validate_utility_report / validate_privacy_report)

## Dependencies
- Plan 4.1 must be complete (recordsets + client stages + interface handlers)

## Tasks

<task type="auto">
  <name>Extend server requires with generation and validation calls</name>
  <files>
    apps/fl_server/src/fl_server/requires.py
  </files>
  <action>
    - Add `generate_synthetic_data(driver: Driver, node_ids: list[int], n_samples: int) -> dict[int, str]`:
      - Use `execution_flow` with `create_generate_synthetic_data_recordset` (MessageType.QUERY, group="generate_synthetic_data").
      - From the replies, extract `reply.content.configs_records["config"]["synthetic_csv"]` per node.
      - Return `{node_id: csv_str}` dict.
    - Add `validate_synthetic_data(driver: Driver, node_ids: list[int], synthetic_csv: str) -> dict[int, dict]`:
      - Use `execution_flow` with `create_validate_synthetic_data_recordset(synthetic_csv)` (MessageType.QUERY, group="validate_synthetic_data").
      - From replies extract `utility_report` and `privacy_report` from `configs_records["config"]`.
      - Return `{node_id: {"utility_report": ..., "privacy_report": ...}}` per node.
  </action>
  <verify>uv run pytest apps/fl_server/tests/ -v</verify>
  <done>Server can request synthetic data generation and collect per-node validation reports.</done>
</task>

<task type="auto">
  <name>Add create_postprocessing_run to mlflow_client</name>
  <files>
    apps/fl_server/src/fl_server/interfaces/mlflow_client.py
  </files>
  <action>
    - Add `create_postprocessing_run(experiment_id: str, parent_run_id: str) -> str`:
      - Call `_mlflow_client.create_run(experiment_id=experiment_id, run_name="postprocessing",
        tags={MLFLOW_PARENT_RUN_ID: parent_run_id, "type": "postprocessing"})`.
      - Return the new run's `run_id`.
    - Add `log_artifact_str(content: str, artifact_path: str, run_id: str) -> None`:
      - Write content to a temporary file, call `_mlflow_client.log_artifact(run_id, local_path, artifact_path)`.
      - Clean up temp file. Use Python `tempfile.NamedTemporaryFile`.
    - Add `finish_run(run_id: str) -> None`:
      - Call `_mlflow_client.update_run(run_id, "FINISHED")`.
  </action>
  <verify>uv run pytest apps/fl_server/tests/ -v</verify>
  <done>MLflow client can open, log artifacts to, and close a sibling postprocessing run.</done>
</task>

<task type="auto">
  <name>Wire post-training block in server_main</name>
  <files>
    apps/fl_server/src/fl_server/app.py
  </files>
  <action>
    - Import `json`, `random`, `pandas as pd`, `io`, `catalogue_hook` from `model_store_interface`.
    - After the "Uploading trained model" block and before "Cleaning up MLflow configuration",
      insert an `if task.dataset_meta_json:` block that:
      1. Deserializes `dataset_meta_json` to `DatasetMeta` using `DatasetMeta.model_validate(json.loads(task.dataset_meta_json))`.
      2. Opens a postprocessing run:
         `post_run_id = mlflow_client.create_postprocessing_run(experiment_id, parent_run_id)`.
         (Retrieve `experiment_id` and `parent_run_id` from `mlflow_client._current_config`.)
      3. Selects a single node for generation:
         `generator_node_id = random.choice(filtered_node_ids)`.
      4. Calls `node_csvs = requires.generate_synthetic_data(driver, [generator_node_id], n_samples=1000)`.
      5. Concatenates all CSVs (just one node) into a single `synthetic_data` DataFrame using `pd.concat`.
      6. Broadcasts synthetic CSV to all nodes via `requires.validate_synthetic_data(driver, filtered_node_ids, list(node_csvs.values())[0])`.
      7. Collects `utility_reports` and `privacy_reports` dicts from the returned per-node results.
      8. Logs each HTML report as an MLflow artifact under `postprocessing/reports/`:
         ```python
         for node_id, reports in validation_results.items():
             mlflow_client.log_artifact_str(reports["utility_report"], f"postprocessing/reports/{node_id}_utility.html", post_run_id)
             mlflow_client.log_artifact_str(reports["privacy_report"], f"postprocessing/reports/{node_id}_privacy.html", post_run_id)
         ```
      9. Logs the synthetic CSV as artifact: `mlflow_client.log_artifact_str(synthetic_csv, "postprocessing/synthetic_data.csv", post_run_id)`.
      10. Calls `catalogue_hook.publish_to_catalogue(synthetic_data, utility_reports, privacy_reports, dataset_meta, experiment_name=task.experiment_name)`.
      11. Closes the postprocessing run: `mlflow_client.finish_run(post_run_id)`.
    - If `task.dataset_meta_json` is None, log a warning: `"dataset_meta_json not provided; skipping post-training validation"`.
  </action>
  <verify>uv run pytest apps/fl_server/tests/ -v</verify>
  <done>server_main orchestrates the full post-training pipeline when metadata is present.</done>
</task>

## Success Criteria
- [ ] Server requests synthetic data from one elected node.
- [ ] Synthetic CSV is broadcast for federated validation.
- [ ] Per-node HTML reports and the CSV are logged as MLflow artifacts on a sibling postprocessing run.
- [ ] `publish_to_catalogue` is called with the consolidated results.
- [ ] If `dataset_meta_json` is None the post-training block is gracefully skipped with a warning.
- [ ] All existing server tests continue to pass.
