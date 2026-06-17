---
phase: 4
plan: 1
wave: 1
---

# Plan 4.1: Extend Recordset Interface & Client Stage Hooks

## Objective
Extend `recordset.py` and `fl_client` to support synthetic data generation and federated post-training validation logic using the `model_store_interface` plugins.

## Context
- .gsd/ROADMAP.md
- common/interfaces/recordset.py
- apps/fl_client/src/fl_client/interface.py
- apps/fl_client/src/fl_client/stages.py
- model_store_interface validation hooks

## Tasks

<task type="auto">
  <name>Extend common recordset configuration</name>
  <files>
    common/interfaces/recordset.py
  </files>
  <action>
    - Add `create_generate_synthetic_data_recordset(n_samples: int) -> RecordSet` using `mode="generate_synthetic_data"`.
    - Add `read_generate_synthetic_data_recordset(recordset: RecordSet) -> int` to extract `n_samples`.
    - Add `create_validate_synthetic_data_recordset(synthetic_csv: str) -> RecordSet` using `mode="validate_synthetic_data"`.
    - Add `read_validate_synthetic_data_recordset(recordset: RecordSet) -> str` to extract `synthetic_csv`.
  </action>
  <verify>uv run pytest common/tests/ -v</verify>
  <done>Recordset exposes creation and reading methods for generate and validate operations.</done>
</task>

<task type="auto">
  <name>Implement client execution stages</name>
  <files>
    apps/fl_client/src/fl_client/stages.py
  </files>
  <action>
    - Import `io` and `pandas as pd`.
    - Import `validate_utility_report` and `validate_privacy_report` from `model_store_interface.validation`.
    - Import `DatasetPartition` from `model_store_interface.core.dataset_partition`.
    - Add `generate_synthetic_data(n_samples: int, global_vars: dict) -> str`:
      - Extract `dataset_meta` and `local_learner` from `global_vars`.
      - Check if `dataset_meta` is present; if not, raise ValueError.
      - Call `df = local_learner.generate_synthetic_data(n_samples, dataset_meta)`.
      - Convert DataFrame to CSV string using `df.to_csv(index=False)` and return.
    - Add `validate_synthetic_data(synthetic_csv: str, global_vars: dict) -> dict[str, str]`:
      - Load CSV string to `synthetic_data` DataFrame using `pd.read_csv(io.StringIO(synthetic_csv))`.
      - Extract `dataset_meta` and `real_data` (`global_vars[DatasetPartition.TEST]`).
      - Check if `dataset_meta` and `real_data` are present.
      - Call `validate_utility_report(synthetic_data, real_data, dataset_meta, node_id=global_vars["node_name"])`.
      - Call `validate_privacy_report(synthetic_data, real_data, dataset_meta, node_id=global_vars["node_name"])`.
      - Return dict `{"utility_report": ..., "privacy_report": ...}`.
  </action>
  <verify>uv run pytest apps/fl_client/tests/ -v</verify>
  <done>Stages are correctly implemented to interact with local learner and validation logic.</done>
</task>

<task type="auto">
  <name>Bind RPC interface to client stages</name>
  <files>
    apps/fl_client/src/fl_client/interface.py
  </files>
  <action>
    - Add `generate_synthetic_data(msg: Message, global_vars: dict) -> Message`:
      - Call `n_samples = recordset.read_generate_synthetic_data_recordset(msg.content)`.
      - `csv_str = stages.generate_synthetic_data(n_samples, global_vars)`.
      - Return success response where `configs_records={"config": ConfigsRecord({"synthetic_csv": csv_str})}`.
    - Add `validate_synthetic_data(msg: Message, global_vars: dict) -> Message`:
      - Call `csv_str = recordset.read_validate_synthetic_data_recordset(msg.content)`.
      - `reports = stages.validate_synthetic_data(csv_str, global_vars)`.
      - Return success response where `configs_records={"config": ConfigsRecord({"utility_report": reports["utility_report"], "privacy_report": reports["privacy_report"]})}`.
  </action>
  <verify>uv run pytest apps/fl_client/tests/ -v</verify>
  <done>Flower messages correctly route to the validation and generation stages.</done>
</task>

## Success Criteria
- [ ] Recordsets for generation and validation exist.
- [ ] Client stages interact with the learner and validation plugins properly.
- [ ] RPC handlers return CSV payloads and HTML reports correctly.
