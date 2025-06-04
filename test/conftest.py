import pytest
import numpy as np


@pytest.fixture
def assert_almost_equal():
    """Use assert_almost equal to 3 significant digits by default"""

    def _assert_almost_equal(actual, desired, sig_digits=3):
        # Convert to float to handle both scalar and array-like inputs
        actual = float(actual)
        desired = float(desired)

        # Calculate the order of magnitude of the number
        if desired == 0:
            order = 0
        else:
            order = int(np.floor(np.log10(abs(desired))))

        # Calculate tolerance based on significant digits and order of magnitude
        atol = 10 ** (-sig_digits + order + 1)

        # Use assert_allclose with relative tolerance
        np.testing.assert_allclose(actual, desired, atol=atol)

    return _assert_almost_equal
