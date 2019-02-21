import unittest
import pandas as pd
from pandas.testing import assert_series_equal
from zenml.utils import array_to_series


class TestPandas(unittest.TestCase):
    def test_array_to_series(self):
        some_array = [1, 2, 3, 4]
        result = pd.Series(some_array, index=list(range(0, len(some_array))))
        assert_series_equal(array_to_series(some_array), result, check_dtype=True)


if __name__ == '__main__':
    unittest.main()
