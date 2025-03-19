import pytest
import numpy as np


@pytest.fixture
def assert_almost_equal():
    """Use assert_almost equal to 4 decimals by default"""

    def _assert_almost_equal(actual, desired, decimal=4):
        np.testing.assert_almost_equal(actual, desired, decimal=decimal)

    return _assert_almost_equal
