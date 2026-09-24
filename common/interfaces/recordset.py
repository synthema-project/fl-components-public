import json
from flwr.common import ConfigsRecord, ParametersRecord, RecordSet


def create_filter_clients_recordset(use_case: str) -> RecordSet:
    configsrecord = ConfigsRecord({"mode": "filter_clients", "use_case": use_case})
    return RecordSet(configs_records={"config": configsrecord})


def read_filter_clients_recordset(recordset: RecordSet) -> str:
    use_case = recordset.configs_records["config"]["use_case"]
    if not isinstance(use_case, str):
        raise TypeError(f"use_case must be a string, received {type(use_case)}")
    return use_case


def create_load_data_recordset(
    use_case: str,
    dataset_paths: list[str] | dict,
    metadata_json: str | None = None,
    evaluation_config_json: str | None = None,
) -> RecordSet:

    configsrecord = ConfigsRecord(
        {
            "mode": "load_data",
            "use_case": use_case,
            "dataset_paths": json.dumps(dataset_paths) if isinstance(dataset_paths, dict) else dataset_paths,
            "metadata_json": metadata_json or "",
            "evaluation_config_json": evaluation_config_json or "",
        }
    )

    return RecordSet(
        configs_records={"config": configsrecord},
    )


def read_load_data_recordset(recordset: RecordSet) -> tuple[str, list[str] | dict, str | None, str | None]:
    config = recordset.configs_records["config"]

    use_case = config["use_case"]
    dataset_paths = config.get("dataset_paths", [])
    if isinstance(dataset_paths, str):
        dataset_paths = json.loads(dataset_paths)
    metadata_json = config.get("metadata_json", "")
    evaluation_config_json = config.get("evaluation_config_json", "")

    if not isinstance(use_case, str):
        raise TypeError(f"use_case must be a string, received {type(use_case)}")

    if not isinstance(dataset_paths, (list, dict)):
        raise TypeError(f"dataset_paths must be a list or role mapping, received {type(dataset_paths)}")

    if not isinstance(metadata_json, str) or not isinstance(evaluation_config_json, str):
        raise TypeError("metadata_json and evaluation_config_json must be strings")

    return use_case, dataset_paths, metadata_json or None, evaluation_config_json or None


def create_prepare_data_recordset() -> RecordSet:
    configsrecord = ConfigsRecord(
        {
            "mode": "prepare_data",
        }
    )
    return RecordSet(
        configs_records={"config": configsrecord},
    )


def create_load_model_recordset() -> RecordSet:
    configsrecord = ConfigsRecord(
        {
            "mode": "load_model",
        }
    )
    return RecordSet(
        configs_records={"config": configsrecord},
    )


def create_upload_model_recordset(parametersrecord: ParametersRecord) -> RecordSet:
    configrecord = ConfigsRecord(
        {
            "mode": "upload_model",
        }
    )
    return RecordSet(
        configs_records={"config": configrecord},
        parameters_records={"parameters": parametersrecord},
    )


def read_upload_model_recordset(recordset: RecordSet) -> ParametersRecord:
    return recordset.parameters_records["parameters"]


def create_generate_synthetic_data_recordset(n_samples: int, pool_id: str) -> RecordSet:
    return RecordSet(configs_records={"config": ConfigsRecord({
        "mode": "generate_synthetic_data",
        "n_samples": n_samples,
        "pool_id": pool_id,
    })})


def create_validate_synthetic_pool_recordset(pool_id: str, object_uri: str,
                                             use_case: str) -> RecordSet:
    return RecordSet(configs_records={"config": ConfigsRecord({
        "mode": "validate_synthetic_pool", "pool_id": pool_id,
        "object_uri": object_uri, "use_case": use_case,
    })})


def read_validate_synthetic_pool_recordset(recordset: RecordSet) -> tuple[str, str, str]:
    config = recordset.configs_records["config"]
    return str(config["pool_id"]), str(config["object_uri"]), str(config["use_case"])


def read_generate_synthetic_data_recordset(recordset: RecordSet) -> int:
    n_samples = recordset.configs_records["config"]["n_samples"]
    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")
    return n_samples


def create_get_parameters_recordset() -> RecordSet:
    configsrecord = ConfigsRecord(
        {
            "mode": "get_parameters",
        }
    )
    return RecordSet(configs_records={"config": configsrecord})


def create_set_parameters_recordset(parametersrecord: ParametersRecord) -> RecordSet:
    configrecord = ConfigsRecord(
        {
            "mode": "set_parameters",
        }
    )
    return RecordSet(
        configs_records={"config": configrecord},
        parameters_records={"parameters": parametersrecord},
    )


def read_set_parameters_recordset(recordset: RecordSet) -> ParametersRecord:
    return recordset.parameters_records["parameters"]


def create_set_run_cfg_recordset(
    experiment_id: str, run_id: str, model_name: str, model_version: int
) -> RecordSet:
    configsrecord = ConfigsRecord(
        {
            "mode": "set_run_config",
            "experiment_id": experiment_id,
            "run_id": run_id,
            "model_name": model_name,
            "model_version": model_version,
        }
    )
    return RecordSet(
        configs_records={"config": configsrecord},
    )

def read_set_run_cfg_recordset(recordset: RecordSet) -> dict:
    experiment_id = recordset.configs_records["config"]["experiment_id"]
    parent_run_id = recordset.configs_records["config"]["run_id"]
    model_name = recordset.configs_records["config"]["model_name"]
    model_version = recordset.configs_records["config"]["model_version"]
    if (
        not isinstance(experiment_id, str)
        or not isinstance(parent_run_id, str)
        or not isinstance(model_name, str)
        or not isinstance(model_version, int)
    ):
        raise TypeError(
            "experiment_id, parent_run_id, model_name must be strings and model_version must be an integer"
        )
    return {
        "experiment_id": experiment_id,
        "parent_run_id": parent_run_id,
        "model_name": model_name,
        "model_version": model_version,
    }

def read_start_P2P_server_recordset(recordset: RecordSet) -> dict:
    port = recordset.configs_records["config"]["port"]
    if not isinstance(port, int):
        raise TypeError(f"port must be an integer, received {type(port)}")
    return {
        "port": port,
    }

def read_discover_peers_recordset(recordset: RecordSet) -> dict:
    min_peers = recordset.configs_records["config"]["min_peers"]
    client_port = recordset.configs_records["config"]["client_port"]
    max_discovery_time = recordset.configs_records["config"]["max_discovery_time"]
    if not isinstance(min_peers, int):
        raise TypeError(f"min_peers must be an integer, received {type(min_peers)}")
    if not isinstance(client_port, int):
        raise TypeError(f"client_port must be an integer, received {type(client_port)}")
    if not isinstance(max_discovery_time, int):
        raise TypeError(f"max_discovery_time must be an integer, received {type(max_discovery_time)}")
    return {
        "min_peers": min_peers,
        "client_port": client_port,
        "max_discovery_time": max_discovery_time
    }
def create_set_dp_cfg_recordset(
    dp_mode: str, clipping_norm: int, noise_multiplier: int
) -> RecordSet:
    configsrecord = ConfigsRecord(
        {
            "mode": "set_dp_config",
            "dp_mode": dp_mode,
            "clipping_norm": clipping_norm,
            "noise_multiplier": noise_multiplier,
        }
    )
    return RecordSet(
        configs_records={"config": configsrecord},
    )

def read_set_dp_cfg_recordset(recordset: RecordSet) -> dict:
    dp_mode = recordset.configs_records["config"]["dp_mode"]
    clipping_norm = recordset.configs_records["config"]["clipping_norm"]
    noise_multiplier = recordset.configs_records["config"]["noise_multiplier"]
    if (
        not isinstance(dp_mode, str)
        or not isinstance(clipping_norm, int)
        or not isinstance(noise_multiplier, int)
    ):
        raise TypeError(
            "dp_mode must be a string and clipping_norm and model version must be integera"
        )
    return {
        "dp_mode": dp_mode,
        "clipping_norm": clipping_norm,
        "noise_multiplier": noise_multiplier,
    }



def create_train_model_recordset(
    parametersrecord: ParametersRecord, current_global_iter: int
) -> RecordSet:
    return RecordSet(
        parameters_records={"parameters": parametersrecord},
        configs_records={
            "config": ConfigsRecord({"current_global_iter": current_global_iter})
        },
    )


def read_train_model_recordset(recordset: RecordSet) -> tuple[ParametersRecord, int]:
    parameters = recordset.parameters_records["parameters"]
    current_global_iter = recordset.configs_records["config"]["current_global_iter"]
    if not isinstance(current_global_iter, int):
        raise TypeError(
            f"current_global_iter must be an integer, received {type(current_global_iter)}"
        )
    return parameters, current_global_iter


def create_clean_config_recordset(status: str = "FINISHED") -> RecordSet:
    configsrecord = ConfigsRecord(
        {
            "mode": "clean_config",
            "status": status,
        }
    )
    return RecordSet(configs_records={"config": configsrecord})


def create_start_P2P_server_recordset(node_ids: list[int], port: int) -> RecordSet:
    configsrecord = ConfigsRecord(
        {
            "mode": "start_P2P_server",
            "node_ids": node_ids,
            "port": port
        }
    )
    return RecordSet(configs_records={"config": configsrecord})

def create_discover_peers_recordset(node_ids: list[int], client_port: int, max_discovery_time: int) -> RecordSet:
    configsrecord = ConfigsRecord(
        {
            "mode": "discover_peers",
            "min_peers": len(node_ids),
            "client_port": client_port,
            "max_discovery_time": max_discovery_time
        }
    )
    return RecordSet(configs_records={"config": configsrecord})

def create_connect_peers_recordset(node_ids: list[int], peers: list[list[str]]) -> RecordSet:
    configsrecord = ConfigsRecord(
        {
            "mode": "connect_peers",
            "node_ids": node_ids,
            "peers": peers # type: ignore
        }
    )
    return RecordSet(configs_records={"config": configsrecord})

def create_set_smpc_enabled_recordset(smpc_enabled: bool) -> RecordSet:
    configsrecord = ConfigsRecord(
        {
            "mode": "set_smpc_enabled",
            "smpc_enabled": smpc_enabled
        }
    )
    return RecordSet(configs_records={"config": configsrecord})

def read_set_smpc_enabled_recordset(recordset: RecordSet) -> bool:
    smpc_enabled = recordset.configs_records["config"]["smpc_enabled"]
    if not isinstance(smpc_enabled, bool):
        raise TypeError(f"smpc_enabled must be a boolean, received {type(smpc_enabled)}")
    return smpc_enabled

def create_generate_synthetic_recordset(
    use_case: str,
    n_samples: int,
    sdg_model_name: str,
    dataset_meta_json: str = "",
    upload_to_s3: bool = True,
    s3_bucket: str = "synthetic-data-for-inference",
    create_s3_bucket_if_missing: bool = False,
    real_data_partition: str = "test",
) -> RecordSet:
    """Create a RecordSet instructing clients to generate synthetic data and upload it."""
    configsrecord = ConfigsRecord(
        {
            "mode": "generate_synthetic",
            "use_case": use_case,
            "n_samples": n_samples,
            "sdg_model_name": sdg_model_name,
            "dataset_meta_json": dataset_meta_json,
            "real_data_partition": real_data_partition,
            "upload_to_s3": upload_to_s3,
            "s3_bucket": s3_bucket,
            "create_s3_bucket_if_missing": create_s3_bucket_if_missing,
        }
    )
    return RecordSet(configs_records={"config": configsrecord})


def read_generate_synthetic_recordset(recordset: RecordSet) -> dict:
    config = recordset.configs_records["config"]
    use_case = config.get("use_case")
    n_samples = config.get("n_samples")
    sdg_model_name = config.get("sdg_model_name")
    dataset_meta_json = config.get("dataset_meta_json", "")
    real_data_partition = config.get("real_data_partition", "test")
    upload_to_s3 = config.get("upload_to_s3", True)
    s3_bucket = config.get("s3_bucket", "synthetic-data-for-inference")
    create_bucket = config.get("create_s3_bucket_if_missing", False)

    if not isinstance(use_case, str):
        raise TypeError(f"use_case must be a string, received {type(use_case)}")
    if not isinstance(n_samples, int):
        raise TypeError(f"n_samples must be an int, received {type(n_samples)}")
    if not isinstance(sdg_model_name, str):
        raise TypeError(f"sdg_model_name must be a string, received {type(sdg_model_name)}")

    return {
        "use_case": use_case,
        "n_samples": n_samples,
        "sdg_model_name": sdg_model_name,
        "dataset_meta_json": dataset_meta_json,
        "real_data_partition": real_data_partition,
        "upload_to_s3": upload_to_s3,
        "s3_bucket": s3_bucket,
        "create_s3_bucket_if_missing": create_bucket,
    }
