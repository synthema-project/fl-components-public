from flwr.common import ConfigsRecord, ParametersRecord, RecordSet


def create_filter_clients_recordset(use_case: str) -> RecordSet:
    configsrecord = ConfigsRecord({"mode": "filter_clients", "use_case": use_case})
    return RecordSet(configs_records={"config": configsrecord})


def read_filter_clients_recordset(recordset: RecordSet) -> str:
    use_case = recordset.configs_records["config"]["use_case"]
    if not isinstance(use_case, str):
        raise TypeError(f"use_case must be a string, received {type(use_case)}")
    return use_case


def create_load_data_recordset(use_case: str) -> RecordSet:
    configsrecord = ConfigsRecord(
        {
            "mode": "load_data",
            "use_case": use_case,
        }
    )
    return RecordSet(
        configs_records={"config": configsrecord},
    )


def read_load_data_recordset(recordset: RecordSet) -> str:
    use_case = recordset.configs_records["config"]["use_case"]
    if not isinstance(use_case, str):
        raise TypeError(f"use_case must be a string, received {type(use_case)}")
    return use_case


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


def create_clean_config_recordset() -> RecordSet:
    configsrecord = ConfigsRecord(
        {
            "mode": "clean_config",
        }
    )
    return RecordSet(configs_records={"config": configsrecord})
