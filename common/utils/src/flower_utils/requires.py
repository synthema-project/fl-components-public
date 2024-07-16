from flwr.common import ConfigsRecord, ParametersRecord, RecordSet


def create_filter_clients_recordset():
    configsrecord = ConfigsRecord({"mode": "filter_clients"})
    return RecordSet(configs_records={"config": configsrecord})


def create_load_data_recordset(use_case: str):
    configsrecord = ConfigsRecord(
        {
            "mode": "load_data",
            "use_case": use_case,
        }
    )
    return RecordSet(
        configs_records={"config": configsrecord},
    )


def create_prepare_data_recordset():
    configsrecord = ConfigsRecord(
        {
            "mode": "prepare_data",
        }
    )
    return RecordSet(
        configs_records={"config": configsrecord},
    )


def create_load_model_recordset(model_name: str, model_version: int):
    configsrecord = ConfigsRecord(
        {
            "mode": "load_model",
            "model_name": model_name,
            "model_version": model_version,
        }
    )
    return RecordSet(
        configs_records={"config": configsrecord},
    )


def create_upload_model_recordset(parametersrecord: ParametersRecord):
    configrecord = ConfigsRecord(
        {
            "mode": "upload_model",
        }
    )
    return RecordSet(
        configs_records={"config": configrecord},
        parameters_records={"parameters": parametersrecord},
    )


def create_get_parameters_recordset():
    configsrecord = ConfigsRecord(
        {
            "mode": "get_parameters",
        }
    )
    return RecordSet(configs_records={"config": configsrecord})


def create_set_parameters_recordset(parametersrecord: ParametersRecord):
    configrecord = ConfigsRecord(
        {
            "mode": "set_parameters",
        }
    )
    return RecordSet(
        configs_records={"config": configrecord},
        parameters_records={"parameters": parametersrecord},
    )


def create_set_run_recordset(experiment_id: int, run_id: int):
    configsrecord = ConfigsRecord(
        {
            "mode": "set_run_config",
            "experiment_id": experiment_id,
            "run_id": run_id,
        }
    )
    return RecordSet(
        configs_records={"config": configsrecord},
    )


def create_train_model_recordset(
    parametersrecord: ParametersRecord, current_global_iter: int
):
    return RecordSet(
        parameters_records={"parameters": parametersrecord},
        configs_records={
            "config": ConfigsRecord({"current_global_iter": current_global_iter})
        },
    )
