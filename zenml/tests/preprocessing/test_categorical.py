import unittest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal
from zenml.preprocessing import count_encode


class TestCategoricsal(unittest.TestCase):
    def test_count_encode(self):
        df = pd.DataFrame({'name': ['Bobby', 'John', 'Bobby', 'Alice']})
        result = pd.DataFrame({'name': [2, 1, 2, 1]})
        assert_frame_equal(count_encode(df), result, check_dtype=True)


if __name__ == '__main__':
    unittest.main()
