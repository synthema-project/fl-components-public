from common.fl_models.iris.utils import Utils


def create_aggregator():
    from collections import OrderedDict

    import numpy as np
    from flwr.common import ParametersRecord

    class Aggregator:
        def aggregate(self, parameter_list: list[ParametersRecord]) -> ParametersRecord:
            parameters = [Utils.parameters_to_dict(param) for param in parameter_list]
            keys = parameters[0].keys()
            result = OrderedDict()
            for key in keys:
                # Init array
                this_array: np.ndarray = np.zeros_like(parameters[0][key])
                for p in parameters:
                    this_array += p[key]
                result[key] = this_array / len(parameter_list)
            return Utils.dict_to_parameter_record(result)

    return Aggregator()
