from collections import OrderedDict

import numpy as np
from flwr.common import Array, NDArray, ParametersRecord


class Utils:
    @staticmethod
    def _basic_array_deserialisation(array: Array) -> NDArray:
        return np.frombuffer(buffer=array.data, dtype=array.dtype).reshape(array.shape)

    @staticmethod
    def parameters_to_dict(params_record: ParametersRecord) -> OrderedDict:
        state_dict = OrderedDict()
        for k, v in params_record.items():
            state_dict[k] = Utils._basic_array_deserialisation(v)

        return state_dict

    @staticmethod
    def _ndarray_to_array(ndarray: NDArray) -> Array:
        """Represent NumPy ndarray as Array."""
        return Array(
            data=ndarray.tobytes(),
            dtype=str(ndarray.dtype),
            stype="numpy.ndarray.tobytes",
            shape=list(ndarray.shape),
        )

    @staticmethod
    def dict_to_parameter_record(
        parameters: OrderedDict["str", NDArray],
    ) -> ParametersRecord:
        state_dict = OrderedDict()
        for k, v in parameters.items():
            state_dict[k] = Utils._ndarray_to_array(v)

        return ParametersRecord(state_dict)

    @staticmethod
    def pytorch_to_parameter_record(
        state_dict: dict,
    ) -> ParametersRecord:
        """Serialise your PyTorch model."""
        transformed_state_dict = OrderedDict()

        for k, v in state_dict.items():
            transformed_state_dict[k] = Utils._ndarray_to_array(v.numpy())

        return ParametersRecord(transformed_state_dict)

    @staticmethod
    def parameters_to_pytorch_state_dict(
        params_record: ParametersRecord,
    ) -> dict:
        # Make sure to import locally torch as it is only available in the server
        import torch

        """Reconstruct PyTorch state_dict from its serialised representation."""
        state_dict = {}
        for k, v in params_record.items():
            state_dict[k] = torch.tensor(Utils._basic_array_deserialisation(v))

        return state_dict
