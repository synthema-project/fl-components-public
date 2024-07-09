import numpy as np
from flwr.common import Array, NDArray


def _ndarray_to_array(ndarray: NDArray) -> Array:
    """Represent NumPy ndarray as Array."""
    return Array(
        data=ndarray.tobytes(),
        dtype=str(ndarray.dtype),
        stype="numpy.ndarray.tobytes",
        shape=list(ndarray.shape),
    )


def _array_to_ndarray(array: Array) -> NDArray:
    return np.frombuffer(buffer=array.data, dtype=array.dtype).reshape(array.shape)
