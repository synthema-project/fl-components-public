from pytest import fixture

import numpy as np

from interfaces.array_serde import array_to_ndarray, ndarray_to_array


@fixture(scope="session")
def ndarray():
    return np.random.rand(10, 10).astype(np.float32)


def test_serde(ndarray):
    array = ndarray_to_array(ndarray)
    deserialized = array_to_ndarray(array)
    assert np.array_equal(ndarray, deserialized)
