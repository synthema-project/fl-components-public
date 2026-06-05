import json
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest
from flwr.common import Message, Metadata, RecordSet, ConfigsRecord
from model_store_interface.core.dataset_partition import DatasetPartition
from model_store_interface.core.schemas.dataset_meta import DatasetMeta
from pytest_bdd import scenario, given, when, then

from fl_client import interface

# --- Constants & Fixtures ---
MOCK_AML_JSON = json.dumps({
    "title": "AML Validation Pool",
    "version": "1.0",
    "schema_version": "dataset_meta_v1",
    "use_case": "AML",
    "variables": []
})


@pytest.fixture
def global_vars():
    """Initialise the isolated state dictionary for the edge node."""
    return {
        "node_name": "ubuntu-edge-01",
        "data_path": "common/data/iris.csv",
        DatasetPartition.TEST: pd.DataFrame({"Age": [1, 2], "HB": [9.1, 10.2]})
    }


# --- Scenario Binding ---
@scenario('../features/aml_clinical_validation.feature', 'Successful Edge Hydration and Federated Validation')
def test_aml_validation_pipeline():
    """The central Pytest-BDD orchestrator."""
    pass


# --- Step Definitions ---
@given('an active Ubuntu edge node holding a 20% real AML hold-out partition')
def active_edge_node(global_vars):
    assert DatasetPartition.TEST in global_vars
    assert not global_vars[DatasetPartition.TEST].empty


@given('a serialised DCAT-AP Data Contract from the FL Server', target_fixture="flower_message")
def serialised_data_contract():
    """Constructs the exact Flower gRPC transport layer payload."""
    configs_records = {
        "config": ConfigsRecord({
            "experiment_id": "exp_001",
            "run_id": "run_001",
            "model_name": "gmm_aml",
            "model_version": 1,
            "dataset_meta_json": MOCK_AML_JSON
        })
    }
    return Message(metadata=MagicMock(spec=Metadata), content=RecordSet(configs_records=configs_records))


@when('the Flower gRPC network triggers the set_run_config instruction')
@patch('fl_client.interface.stages.set_run_config')
def trigger_set_run_config(mock_set_run_config, flower_message, global_vars):
    interface.set_run_config_if(flower_message, global_vars)


@then('the edge node mathematically hydrates the Pydantic DatasetMeta object')
def verify_edge_hydration(global_vars):
    """Eradicating Primitive Obsession: Verifying the object is strongly typed."""
    meta = global_vars.get("dataset_meta")
    assert isinstance(meta, DatasetMeta), "Primitive Obsession detected! Payload is not a Pydantic object."
    assert meta.use_case.value == "AML", "Schema drift: Use case misalignment."


@when('the server requests distributed synthetic data validation', target_fixture="validation_message")
def server_requests_validation():
    configs_records = {
        "config": ConfigsRecord({
            "mode": "validate_synthetic_data",
            "synthetic_csv": "Age,HB\n42,8.5\n55,11.0"
        })
    }
    return Message(metadata=MagicMock(spec=Metadata), content=RecordSet(configs_records=configs_records))


@then('the node successfully executes the clinical validation plugins')
@patch('fl_client.stages.validate_utility_report', return_value="<html>Utility OK</html>")
@patch('fl_client.stages.validate_privacy_report', return_value="<html>Privacy OK</html>")
def execute_validation_plugins(mock_privacy, mock_utility, validation_message, global_vars):
    response = interface.validate_synthetic_data_if(validation_message, global_vars)
    # Store response inside global_vars to pass to the final assertion
    global_vars["last_response"] = response


@then('generates the HTML privacy and utility reports')
def verify_html_reports(global_vars):
    response = global_vars["last_response"]
    # Verify the Flower RPC response correctly packed the HTML strings
    assert "utility" in response.content.configs_records["config"]
    assert "privacy" in response.content.configs_records["config"]
    assert response.content.configs_records["config"]["utility"] == "<html>Utility OK</html>"
