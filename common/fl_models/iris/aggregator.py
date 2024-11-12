from .utils import Utils


def create_aggregator():
    from collections import OrderedDict

    import numpy as np
    from flwr.common import ParametersRecord, MetricsRecord

    class Aggregator:
        def aggregate_parameters(
            self, parameter_list: list[ParametersRecord]
        ) -> ParametersRecord:
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

        def aggregate_metrics(self, metrics_list: list[MetricsRecord]) -> MetricsRecord:
            keys = metrics_list[0].keys()
            result = OrderedDict()
            for key in keys:
                # Init array
                cumsum = 0.0
                for m in metrics_list:
                    if not isinstance(m[key], (int, float)):
                        raise ValueError(
                            f"MetricsRecord value type not supported: {type(m[key])}"
                        )
                    cumsum += m[key]  # type: ignore
                result[key] = cumsum / len(metrics_list)
            return MetricsRecord(result)  # type: ignore

    return Aggregator()
