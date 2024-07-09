from torch import nn

from common.utils.src.ml_integrations.pytorch import (
    pytorch_to_parameter_record,
    parameters_to_pytorch_state_dict,
)


def test_pytorch_serde():
    model = nn.Linear(10, 1)
    params_record = pytorch_to_parameter_record(model)
    state_dict = parameters_to_pytorch_state_dict(params_record)
    for k, v in model.state_dict().items():
        assert v.numpy().all() == state_dict[k].numpy().all()
