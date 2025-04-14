from typing import Dict, List, Tuple
import numpy as np
import logging
import grpc

logger = logging.getLogger(__name__)
def serialize_parameters(parameters):
    """
    Serialize parameters dict (already Array objects) into a dict of raw fields.
    
    Args:
        parameters (Dict[str, Array]): Input parameters.
    
    Returns:
        Dict[str, Any]: Serialized dictionary.
    """
    serialized = {}
    for name, array_obj in parameters.items():
        serialized[name] = {
            "dtype": array_obj.dtype,
            "shape": array_obj.shape,
            "stype": array_obj.stype,
            "data": array_obj.data
        }
    return serialized


def deserialize_parameters(serialized):
    """
    Deserialize serialized dict into dict of numpy arrays.
    
    Args:
        serialized (Dict[str, Any]): Serialized parameters.
    
    Returns:
        Dict[str, np.ndarray]: Dict of numpy arrays.
    """
    parameters = {}
    for name, info in serialized.items():
        array = np.frombuffer(info["data"], dtype=np.dtype(info["dtype"])).reshape(info["shape"])
        parameters[name] = array
    return parameters

def generate_value_secret_share(value: float, num_shares: int = 3) -> List[np.ndarray]:
    """
    Generates secret shares of a given value using additive secret sharing.

    This method splits a value into `num_shares` parts, where the sum of all shares equals the original value.
    The shares are randomly generated, except for the last share, which is determined so that the sum of 
    all shares equals the original value.

    Parameters:
    value (float): The value to be secret-shared.
    num_shares (int): The number of shares to split the value into. Default is 3.

    Returns:
    list: A list of secret shares (floating-point numbers).
    """
    shares = [np.random.uniform(-1, 1) for _ in range(num_shares - 1)]
    last_share = value - sum(shares)
    shares.append(last_share)
    return shares

def generate_matrix_secret_shares(matrix: np.ndarray, num_shares: int = 3) -> List[np.ndarray]:
    """
    Generates secret shares for each element in a matrix using additive secret sharing.

    This method splits each element in the provided matrix into `num_shares` parts, where the sum of 
    all shares for each element equals the original value. The shares for each element are calculated 
    using the `generate_value_secret_share` method, and the result is a matrix of secret shares where 
    each element in the matrix has been split into shares.

    Parameters:
    matrix (numpy.ndarray): The matrix whose elements are to be secret-shared.
    num_shares (int): The number of shares to split each element into. Default is 3.

    Returns:
    list: A list of matrices, each containing one set of secret shares for each element in the original matrix.
    """
    shares = [np.zeros_like(matrix) for _ in range(num_shares)]
    for index, value in np.ndenumerate(matrix):
        value_shares = generate_value_secret_share(value, num_shares)
        for i in range(num_shares):
            shares[i][index] = value_shares[i]
    return shares

def split_model_parameters_to_shares(
    parameters: Dict[str, np.ndarray],
    node_names: List[str],
    self_node_name: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
    """
    Split parameters into shares for each node.

    Args:
        parameters (dict): Model parameters {name: np.ndarray}.
        node_names (list): List of node names (strings).
        self_node_name (str): Name of the current node.

    Returns:
        local_shares (dict): {param_name: np.ndarray} for this client.
        remote_shares (dict): {node_name: {param_name: np.ndarray}} for other nodes.
    """
    num_clients = len(node_names)

    # Step 1: split each parameter into shares
    shares_per_parameter = {
        name: generate_matrix_secret_shares(param, num_clients)
        for name, param in parameters.items()
    }

    # Step 2: assign shares to each node
    node_shares = {
        node_name: {
            param_name: shares[node_idx]
            for param_name, shares in shares_per_parameter.items()
        }
        for node_idx, node_name in enumerate(node_names)
    }

    # Step 3: split into local and remote
    local_shares = node_shares.pop(self_node_name)  # Take the share of this client
    remote_shares = node_shares                    # Remaining shares for others

    return local_shares, remote_shares