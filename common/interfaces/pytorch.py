from collections import OrderedDict

import torch
from flwr.common import ParametersRecord

from interfaces.array_serde import array_to_ndarray, ndarray_to_array


def pytorch_to_parameter_record(pytorch_module: torch.nn.Module) -> ParametersRecord:
    """Serialise your PyTorch model."""
    raw_state_dict = pytorch_module.state_dict()
    transformed_state_dict = OrderedDict()

    for k, v in raw_state_dict.items():
        transformed_state_dict[k] = ndarray_to_array(v.numpy())

    return ParametersRecord(transformed_state_dict)


def parameters_to_pytorch_state_dict(
    params_record: ParametersRecord,
) -> dict[str, torch.Tensor]:
    """Reconstruct PyTorch state_dict from its serialised representation."""
    state_dict = {}
    for k, v in params_record.items():
        state_dict[k] = torch.tensor(array_to_ndarray(v))

    return state_dict
