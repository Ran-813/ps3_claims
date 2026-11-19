import numpy as np
import pytest

from ps3.preprocessing import Winsorizer

# TODO: Test your implementation of a simple Winsorizer


@pytest.mark.parametrize(
    "lower_quantile, upper_quantile", [(0, 1), (0.05, 0.95), (0.5, 0.5)]
)
def test_winsorizer(lower_quantile, upper_quantile):

    X = np.random.normal(0, 1, 1000)

    # using the winsorizer transformer on the random data
    winsorizer = Winsorizer(lower_quantile, upper_quantile)
    winsorizer.fit(X)
    result = winsorizer.transform(X)

    # check that the the winsorizer works as expected
    # first, we calculate what the limits should be without reading the limits stored by the winsorizer
    lower_limit = np.quantile(X, lower_quantile)
    upper_limit = np.quantile(X, upper_quantile)

    # create copy of original data and replace values outside of the limits with the limits
    expected = X.copy()
    expected[X < lower_limit] = lower_limit
    expected[X > upper_limit] = upper_limit

    # comparing actual results with expected results
    np.testing.assert_allclose(winsorizer.lower_quantile_, lower_limit)
    np.testing.assert_allclose(winsorizer.upper_quantile_, upper_limit)
    np.testing.assert_allclose(result, expected)
    # checks that every transformed value matched the expected values we calculated above
