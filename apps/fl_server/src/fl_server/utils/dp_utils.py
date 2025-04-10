from collections import OrderedDict
from fl_models.iris.utils import Utils
import numpy as np
from flwr.common import ParametersRecord

def add_gaussian_noise_global_parameters(params_record: ParametersRecord, noise_multiplier:int, clipping_norm: int, number_of_clients: int):

    params = Utils.parameters_to_dict(params_record)


    std_dev = noise_multiplier*clipping_norm / number_of_clients
            
    noisy_params = OrderedDict()

    for key, value in params.items():
        noise: np.narray = np.random.normal(0, std_dev, value.shape)
        noisy_value = value + noise
        noisy_params[key] = noisy_value

    noisy_params_record = Utils.dict_to_parameter_record(noisy_params)

    return noisy_params_record
