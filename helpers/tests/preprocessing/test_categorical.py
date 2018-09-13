import unittest
import pandas as pd
from pandas.testing import assert_frame_equal
from helpers.preprocessing import create_dummy_variables


class TestCategorical(unittest.TestCase):
    def test_create_dummy_variables(self):
        df = pd.DataFrame({'eye_color': ['blue', 'green', 'brown', 'blue']})
        prefix = 'eye_color'
        result = pd.DataFrame({'eye_color_blue': [1, 0, 0, 1], 'eye_color_green': [0, 1, 0, 0]})
        assert_frame_equal(create_dummy_variables(df.eye_color, prefix), result, check_dtype=True)


if __name__ == '__main__':
    unittest.main()
