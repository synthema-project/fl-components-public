# ROADMAP.md

> **Current Phase**: Not started
> **Milestone**: v1.0 — MSI Alignment

## Must-Haves (from SPEC)
- [ ] `model-store-interface` bumped to v0.5.0 in all sub-packages (**blocked** — pending PyPI publication, see ADR-002).
- [ ] `train/val/test` node partitions wired to correct FL stages.
- [ ] `HealthDcatApDatasetMeta` accepted (optional, backward-compatible — ADR-001) and propagated through every task.
- [ ] New MLflow run created post-training for synthetic data and validation outputs.
- [ ] `publish_to_catalogue` called from FL server finalisation stage.

---

## Phases

### Phase 1: Dependency & Build Alignment
**Status**: 🚫 BLOCKED — awaiting `model-store-interface` v0.5.0 publication to `pypi.synthema.rid-intrasoft.eu`
**Workaround**: Use `pip install -e path/to/app-model_store-interface` locally while Phases 2–5 are developed (see ADR-002).
**Objective**: Once v0.5.0 is published, bump the version pin in all `pyproject.toml` files, regenerate lock files, and run the full test suite.
**Requirements**: REQ-01
**Files**: `common/pyproject.toml`, `apps/fl_server/pyproject.toml`, `apps/fl_client/pyproject.toml`, `e2e/pyproject.toml`

---

### Phase 2: Dataset Partitioning Protocol
**Status**: ✅ Complete
**Objective**: Introduce `DatasetPartition`-aware data loading in `fl_client`. Each node must split its local dataset into `TRAIN` (60 %), `VAL` (20 %), `TEST` (20 %) using the enum from `model_store_interface.src.dataset_partition`. Update `stages.load_data` to return/store all three splits. Wire `TRAIN` to `train_model`, `VAL` to `evaluate_model` (intra-round), and `TEST` to the post-training federated validation phase.
**Requirements**: REQ-02
**Files**: `apps/fl_client/src/fl_client/stages.py`, `apps/fl_client/src/fl_client/app.py`

---

### Phase 3: HealthDCAT-AP Metadata Propagation
**Status**: ✅ Complete
**Objective**: Add `dataset_meta_json: Optional[str] = None` to `TaskBase` / `TaskCreate` / `TaskRead` (serialized `HealthDcatApDatasetMeta` — optional per ADR-001, backward-compatible). Pass it from the REST API dispatcher through RabbitMQ, into the FL server task, and down to the client via `ConfigsRecord` in the Flower message. The client deserializes it and makes it available for validation and publication calls. When `None`, skip post-training validation with a warning log.
**Requirements**: REQ-03
**Files**: `apps/fl_server/src/fl_server/schemas/task.py`, `apps/fl_server/src/fl_server/app.py`, `apps/fl_client/src/fl_client/stages.py`, `apps/restapi/...`

---

### Phase 4: Post-Training MLflow Run + Federated Validation
**Status**: ⬜ Not Started
**Objective**: Add a post-training block in `fl_server/app.py` (`server_main`) that:
  1. Opens a new sibling MLflow run under the original experiment (using `mlflow_client.create_child_run` with a `postprocessing` tag).
  2. Selects a node for synthetic data generation and requests it via a `QUERY` Flower message.
  3. Broadcasts the generated synthetic data to all nodes.
  4. Collects per-node utility and privacy HTML validation reports.
  5. Logs reports and the synthetic CSV as MLflow artifacts on the new run.
  6. Calls `catalogue_hook.publish_to_catalogue(synthetic_data, reports, dataset_meta, ...)` from `model_store_interface`.
  7. Closes the post-training run.
**Requirements**: REQ-04, REQ-05
**Files**: `apps/fl_server/src/fl_server/app.py`, `apps/fl_server/src/fl_server/interfaces/mlflow_client.py`

---

### Phase 5: MLflow Experiment Auto-Creation & Cleanup
**Status**: ⬜ Not Started
**Objective**: Align `create_parent_run` in both `common/interfaces/mlflow_client.py` and `apps/fl_server/src/fl_server/interfaces/mlflow_client.py` to auto-create the experiment if it does not exist (matching MSI's `submit_fl_model` behaviour), instead of raising `ValueError`. Eliminate code duplication between `common` and `fl_server` MLflow clients.
**Requirements**: REQ-06
**Files**: `common/interfaces/mlflow_client.py`, `apps/fl_server/src/fl_server/interfaces/mlflow_client.py`

---

### Phase 6: Tests & Documentation
**Status**: ⬜ Not Started
**Objective**: Write unit and integration tests for all changed modules. Update `README.md` and inline docstrings. Ensure `ruff` and `mypy` pass cleanly.
**Requirements**: REQ-07
**Files**: `apps/fl_server/tests/`, `apps/fl_client/tests/`, `common/interfaces/tests/`
