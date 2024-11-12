from collections import OrderedDict

import torch
from flwr.common import ParametersRecord

from interfaces.array_serde import array_to_ndarray, ndarray_to_array


def pytorch_to_parameter_record(pytorch_module: torch.nn.Module) -> ParametersRecord:
    """
    Serialize a PyTorch model's state_dict into a ParametersRecord.

    Args:
        pytorch_module (torch.nn.Module): The PyTorch model to be serialized.

    Returns:
        ParametersRecord: A ParametersRecord containing the serialized state_dict of the PyTorch model.
    """
    raw_state_dict = pytorch_module.state_dict()
    transformed_state_dict = OrderedDict()

    for k, v in raw_state_dict.items():
        transformed_state_dict[k] = ndarray_to_array(v.numpy())

    return ParametersRecord(transformed_state_dict)


def parameters_to_pytorch_state_dict(
    params_record: ParametersRecord,
) -> dict[str, torch.Tensor]:
    """
    Reconstruct a PyTorch model's state_dict from a ParametersRecord.

    Args:
        params_record (ParametersRecord): The ParametersRecord containing the serialized state_dict.

    Returns:
        dict[str, torch.Tensor]: A dictionary mapping each parameter name to its corresponding PyTorch tensor.
    """
    state_dict = {}
    for k, v in params_record.items():
        state_dict[k] = torch.tensor(array_to_ndarray(v))

    return state_dict
