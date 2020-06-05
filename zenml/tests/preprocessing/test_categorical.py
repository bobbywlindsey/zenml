"""
Test categorical transformations
"""

import unittest
from pandas.testing import assert_frame_equal
import pandas as pd
from zenml.preprocessing import count_encode


class TestCategorical(unittest.TestCase):
    """
    Test cases for categorical transformations 
    """
    def test_count_encode(self):
        """
        Test count encoding which replaces the categorical
        value with the number of times it occurs in the
        data set
        """
        df = pd.DataFrame({'name': ['Bobby', 'John', 'Bobby', 'Alice']})
        result = pd.DataFrame({'name': [2, 1, 2, 1]})
        assert_frame_equal(count_encode(df), result, check_dtype=True)


if __name__ == '__main__':
    unittest.main()
