from pytest import fixture

import numpy as np

from utils.src.flower_utils.serde import _array_to_ndarray, _ndarray_to_array


@fixture(scope="session")
def ndarray():
    return np.random.rand(10, 10).astype(np.float32)


def test_serde(ndarray):
    array = _ndarray_to_array(ndarray)
    deserialized = _array_to_ndarray(array)
    assert np.array_equal(ndarray, deserialized)
